import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import cudf  # type: ignore

    RAPIDS_AVAILABLE = True
except Exception:
    cudf = None
    RAPIDS_AVAILABLE = False


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


def generate_synthetic_transactions(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tx_amount = rng.lognormal(mean=3.3, sigma=1.0, size=n_rows)
    customer_age = rng.integers(18, 85, size=n_rows)
    hour = rng.integers(0, 24, size=n_rows)
    terminal_risk = rng.uniform(0.0, 1.0, size=n_rows)
    distance_from_home = rng.gamma(shape=2.0, scale=9.0, size=n_rows)
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
    is_fraud = rng.binomial(1, np.clip(fraud_prob, 0.001, 0.999), size=n_rows)

    return pd.DataFrame(
        {
            "tx_amount": tx_amount,
            "customer_age": customer_age,
            "hour": hour,
            "terminal_risk": terminal_risk,
            "distance_from_home": distance_from_home,
            "velocity_1h": velocity_1h,
            "country_mismatch": country_mismatch,
            "is_fraud": is_fraud,
        }
    )


def feature_engineer(df: pd.DataFrame, use_rapids: bool):
    if use_rapids and RAPIDS_AVAILABLE:
        gdf = cudf.DataFrame.from_pandas(df)
        gdf["amount_log"] = np.log1p(gdf["tx_amount"])
        gdf["night_tx"] = ((gdf["hour"] <= 5) | (gdf["hour"] >= 22)).astype("int8")
        gdf["amount_x_risk"] = gdf["tx_amount"] * gdf["terminal_risk"]
        return gdf

    out = df.copy()
    out["amount_log"] = np.log1p(out["tx_amount"])
    out["night_tx"] = ((out["hour"] <= 5) | (out["hour"] >= 22)).astype("int8")
    out["amount_x_risk"] = out["tx_amount"] * out["terminal_risk"]
    return out


def to_numpy_features_labels(df_engineered):
    if RAPIDS_AVAILABLE and hasattr(df_engineered, "to_pandas"):
        pdf = df_engineered.to_pandas()
    else:
        pdf = df_engineered

    x = pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = pdf["is_fraud"].to_numpy(dtype=np.int32)
    return x, y, pdf


def train_and_evaluate(x: np.ndarray, y: np.ndarray, use_gpu: bool, seed: int):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 8,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": seed,
    }

    # XGBoost 2.x prefers device selection over legacy gpu_hist.
    params["device"] = "cuda" if use_gpu else "cpu"

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=180,
        evals=[(dtrain, "train"), (dtest, "valid")],
        verbose_eval=False,
    )

    pred = booster.predict(dtest)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, pred)),
        "test_rows": int(y_test.shape[0]),
        "fraud_rate_test": float(y_test.mean()),
    }
    return booster, metrics


def persist_artifacts(out_dir: Path, booster, metrics: dict, engineered_pdf: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "fraud_xgb_model.json"
    metrics_path = out_dir / "metrics.json"
    data_path = out_dir / "engineered_transactions.parquet"

    booster.save_model(model_path.as_posix())
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    engineered_pdf.to_parquet(data_path, index=False)

    return model_path, metrics_path, data_path


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated fraud workflow with RAPIDS + XGBoost")
    parser.add_argument("--rows", type=int, default=50000, help="Number of synthetic transactions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for model and metrics",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable RAPIDS and GPU training even if available",
    )
    args = parser.parse_args()

    use_rapids = RAPIDS_AVAILABLE and not args.force_cpu
    use_gpu_for_training = use_rapids and not args.force_cpu

    raw = generate_synthetic_transactions(args.rows, args.seed)
    engineered = feature_engineer(raw, use_rapids=use_rapids)
    x, y, engineered_pdf = to_numpy_features_labels(engineered)

    booster, metrics = train_and_evaluate(
        x,
        y,
        use_gpu=use_gpu_for_training,
        seed=args.seed,
    )

    model_path, metrics_path, data_path = persist_artifacts(
        args.output_dir, booster, metrics, engineered_pdf
    )

    runtime_mode = "GPU (RAPIDS + CUDA)" if use_gpu_for_training else "CPU fallback"
    print(f"Run mode: {runtime_mode}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved engineered data: {data_path}")


if __name__ == "__main__":
    main()
