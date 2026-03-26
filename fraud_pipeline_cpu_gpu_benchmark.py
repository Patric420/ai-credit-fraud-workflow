import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import cudf  # type: ignore

    CUDF_AVAILABLE = True
except Exception:
    cudf = None
    CUDF_AVAILABLE = False

try:
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

try:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression  # type: ignore
    from cuml.model_selection import train_test_split as cuml_train_test_split  # type: ignore

    CUML_AVAILABLE = True
except Exception:
    CuMLLogisticRegression = None
    cuml_train_test_split = None
    CUML_AVAILABLE = False

FEATURE_COLUMNS = [
    "tx_amount",
    "customer_age",
    "hour",
    "terminal_risk",
    "distance_from_home",
    "velocity_1h",
    "country_mismatch",
    "amount_log",
    "night_tx",
    "amount_x_risk",
]


class StepTimer:
    def __init__(self, live: bool = True, sync_hook: Optional[Callable[[], None]] = None):
        self.live = live
        self.sync_hook = sync_hook
        self.timings: Dict[str, float] = {}

    def measure(self, step_name: str):
        class _Context:
            def __init__(self, outer: "StepTimer", name: str):
                self.outer = outer
                self.name = name
                self.start = 0.0

            def __enter__(self):
                if self.outer.sync_hook is not None:
                    self.outer.sync_hook()
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.outer.sync_hook is not None:
                    self.outer.sync_hook()
                elapsed = time.perf_counter() - self.start
                self.outer.timings[self.name] = elapsed
                if self.outer.live:
                    print(f"[{self.name}] {elapsed:.6f} sec")

        return _Context(self, step_name)


def generate_synthetic_transactions(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tx_amount = rng.lognormal(mean=3.3, sigma=1.0, size=n_rows).astype(np.float32)
    customer_age = rng.integers(18, 85, size=n_rows)
    hour = rng.integers(0, 24, size=n_rows)
    terminal_risk = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)
    distance_from_home = rng.gamma(shape=2.0, scale=9.0, size=n_rows).astype(np.float32)
    velocity_1h = rng.poisson(lam=3.2, size=n_rows)
    country_mismatch = rng.binomial(1, p=0.06, size=n_rows)

    fraud_logit = (
        -4.4
        + 0.016 * tx_amount
        + 1.8 * terminal_risk
        + 0.72 * country_mismatch
        + 0.06 * np.maximum(hour - 21, 0)
        + 0.05 * velocity_1h
        + 0.004 * distance_from_home
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    is_fraud = rng.binomial(1, np.clip(fraud_prob, 0.001, 0.999), size=n_rows).astype(np.int8)

    return pd.DataFrame(
        {
            "tx_amount": tx_amount,
            "customer_age": customer_age.astype(np.int16),
            "hour": hour.astype(np.int8),
            "terminal_risk": terminal_risk,
            "distance_from_home": distance_from_home,
            "velocity_1h": velocity_1h.astype(np.int16),
            "country_mismatch": country_mismatch.astype(np.int8),
            "is_fraud": is_fraud,
        }
    )


def gpu_sync() -> None:
    if CUPY_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()


def create_cuml_model(max_iter: int):
    try:
        return CuMLLogisticRegression(max_iter=max_iter, output_type="cupy")
    except TypeError:
        return CuMLLogisticRegression(max_iter=max_iter)


def feature_engineering_pandas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_log"] = np.log1p(out["tx_amount"])
    out["night_tx"] = ((out["hour"] <= 5) | (out["hour"] >= 22)).astype("int8")
    out["amount_x_risk"] = out["tx_amount"] * out["terminal_risk"]
    return out


def feature_engineering_cudf(gdf):
    out = gdf.copy(deep=True)
    out["amount_log"] = np.log1p(out["tx_amount"])
    out["night_tx"] = ((out["hour"] <= 5) | (out["hour"] >= 22)).astype("int8")
    out["amount_x_risk"] = out["tx_amount"] * out["terminal_risk"]
    return out


def run_cpu_pipeline(df: pd.DataFrame, seed: int, live: bool, max_iter: int) -> Dict:
    timer = StepTimer(live=live)

    with timer.measure("cpu_feature_engineering"):
        engineered = feature_engineering_pandas(df)

    x = engineered[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = engineered["is_fraud"].to_numpy(dtype=np.int32)

    with timer.measure("cpu_train_test_split"):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed, stratify=y
        )

    with timer.measure("cpu_train"):
        model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        model.fit(x_train, y_train)

    with timer.measure("cpu_inference_eval"):
        pred = model.predict_proba(x_test)[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, pred)),
            "pr_auc": float(average_precision_score(y_test, pred)),
            "test_rows": int(y_test.shape[0]),
            "fraud_rate_test": float(y_test.mean()),
        }

    total = sum(timer.timings.values())
    timer.timings["cpu_end_to_end"] = total
    return {
        "framework": "pandas+scikit-learn",
        "timings_seconds": timer.timings,
        "metrics": metrics,
    }


def run_gpu_pipeline(df: pd.DataFrame, seed: int, live: bool, max_iter: int, warmup: bool) -> Dict:
    if not CUDF_AVAILABLE:
        return {
            "framework": "cuDF+cuML",
            "available": False,
            "reason": "cuDF is not installed or not importable in this environment.",
        }

    if not CUML_AVAILABLE:
        return {
            "framework": "cuDF+cuML",
            "available": False,
            "reason": "cuML is not installed or not importable in this environment.",
        }

    timer = StepTimer(live=live, sync_hook=gpu_sync)

    with timer.measure("gpu_to_cudf"):
        # cuDF API varies by version. Handle both old and new conversion paths.
        if hasattr(cudf.DataFrame, "from_pandas"):
            gdf = cudf.DataFrame.from_pandas(df)
        elif hasattr(cudf, "from_pandas"):
            gdf = cudf.from_pandas(df)
        else:
            gdf = cudf.DataFrame(df)

    with timer.measure("gpu_feature_engineering"):
        engineered = feature_engineering_cudf(gdf)

    x = engineered[FEATURE_COLUMNS]
    y = engineered["is_fraud"].astype("int32")

    with timer.measure("gpu_train_test_split"):
        x_train, x_test, y_train, y_test = cuml_train_test_split(
            x, y, test_size=0.2, random_state=seed, stratify=y
        )

    if warmup:
        warm_rows = min(20000, int(len(y_train)))
        if warm_rows > 0:
            x_warm = x_train.head(warm_rows)
            y_warm = y_train.head(warm_rows)
            warm_model = create_cuml_model(max_iter=40)
            warm_model.fit(x_warm, y_warm)
            _ = warm_model.predict(x_warm)
            gpu_sync()

    with timer.measure("gpu_train"):
        model = create_cuml_model(max_iter=max_iter)
        model.fit(x_train, y_train)

    with timer.measure("gpu_inference_eval"):
        # For ranking metrics (ROC AUC / PR AUC), decision scores are enough and
        # often faster than generating full probability matrices.
        if hasattr(model, "decision_function"):
            decision_scores = model.decision_function(x_test)
            if CUPY_AVAILABLE and isinstance(decision_scores, cp.ndarray):
                pred = cp.asnumpy(decision_scores)
            elif hasattr(decision_scores, "to_numpy"):
                pred = decision_scores.to_numpy()
            else:
                pred = np.asarray(decision_scores)
        else:
            pred_proba = model.predict_proba(x_test)
            if CUPY_AVAILABLE and isinstance(pred_proba, cp.ndarray):
                pred = cp.asnumpy(pred_proba[:, 1])
            elif hasattr(pred_proba, "iloc"):
                pred = pred_proba.iloc[:, 1].to_numpy()
            else:
                pred = np.asarray(pred_proba)[:, 1]

        if CUPY_AVAILABLE and isinstance(y_test, cp.ndarray):
            y_test_np = cp.asnumpy(y_test)
        else:
            y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test_np, pred)),
            "pr_auc": float(average_precision_score(y_test_np, pred)),
            "test_rows": int(y_test_np.shape[0]),
            "fraud_rate_test": float(y_test_np.mean()),
        }

    total = sum(timer.timings.values())
    timer.timings["gpu_end_to_end"] = total
    return {
        "framework": "cuDF+cuML",
        "available": True,
        "timings_seconds": timer.timings,
        "metrics": metrics,
    }


def compute_speedups(cpu_result: Dict, gpu_result: Dict) -> Dict:
    if not gpu_result.get("available", False):
        return {
            "available": False,
            "reason": gpu_result.get("reason", "GPU pipeline unavailable."),
        }

    pairs: Tuple[Tuple[str, str, str], ...] = (
        ("feature_engineering", "cpu_feature_engineering", "gpu_feature_engineering"),
        ("train_test_split", "cpu_train_test_split", "gpu_train_test_split"),
        ("train", "cpu_train", "gpu_train"),
        ("inference_eval", "cpu_inference_eval", "gpu_inference_eval"),
        ("end_to_end", "cpu_end_to_end", "gpu_end_to_end"),
    )

    speedups = {}
    for label, cpu_key, gpu_key in pairs:
        cpu_t = cpu_result["timings_seconds"][cpu_key]
        gpu_t = gpu_result["timings_seconds"][gpu_key]
        speedups[label] = {
            "cpu_seconds": cpu_t,
            "gpu_seconds": gpu_t,
            "speedup_x": (cpu_t / gpu_t) if gpu_t > 0 else None,
        }
    return {"available": True, "by_step": speedups}


def print_summary(cpu_result: Dict, gpu_result: Dict, speedups: Dict) -> None:
    print("\n=== CPU Pipeline (pandas + scikit-learn) ===")
    print(json.dumps(cpu_result["metrics"], indent=2))

    print("\n=== GPU Pipeline (cuDF + cuML) ===")
    if gpu_result.get("available", False):
        print(json.dumps(gpu_result["metrics"], indent=2))
    else:
        print(f"Skipped: {gpu_result.get('reason', 'Unknown reason')}")

    print("\n=== Benchmark: Step Timings & Speedups ===")
    if not speedups.get("available", False):
        print(f"Speedup unavailable: {speedups.get('reason', 'Unknown reason')}")
        return

    header = f"{'Step':<22}{'CPU (s)':>12}{'GPU (s)':>12}{'Speedup':>12}"
    print(header)
    print("-" * len(header))
    for step, stats in speedups["by_step"].items():
        speedup_text = f"{stats['speedup_x']:.2f}x" if stats["speedup_x"] is not None else "n/a"
        print(
            f"{step:<22}{stats['cpu_seconds']:>12.6f}{stats['gpu_seconds']:>12.6f}{speedup_text:>12}"
        )


def save_report(output_dir: Path, report: Dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "cpu_gpu_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fraud detection pipeline on CPU and GPU, then benchmark each step."
    )
    parser.add_argument("--rows", type=int, default=200000, help="Number of synthetic rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Max iterations for LogisticRegression in both CPU and GPU runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where benchmark report is written",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-step live timing output",
    )
    parser.add_argument(
        "--disable-gpu-warmup",
        action="store_true",
        help="Disable short untimed GPU warmup pass before timed GPU steps",
    )
    args = parser.parse_args()

    timer = StepTimer(live=not args.quiet)
    with timer.measure("generate_data"):
        df = generate_synthetic_transactions(args.rows, args.seed)

    cpu_result = run_cpu_pipeline(
        df,
        seed=args.seed,
        live=not args.quiet,
        max_iter=args.max_iter,
    )
    gpu_result = run_gpu_pipeline(
        df,
        seed=args.seed,
        live=not args.quiet,
        max_iter=args.max_iter,
        warmup=not args.disable_gpu_warmup,
    )
    speedups = compute_speedups(cpu_result, gpu_result)

    report = {
        "rows": args.rows,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "gpu_warmup": not args.disable_gpu_warmup,
        "data_generation_seconds": timer.timings["generate_data"],
        "cpu": cpu_result,
        "gpu": gpu_result,
        "speedups": speedups,
    }

    report_path = save_report(args.output_dir, report)
    print_summary(cpu_result, gpu_result, speedups)
    print(f"\nSaved benchmark report: {report_path}")


if __name__ == "__main__":
    main()
