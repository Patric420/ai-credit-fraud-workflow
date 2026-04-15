import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import cudf  # type: ignore

    RAPIDS_AVAILABLE = True
except Exception:
    cudf = None
    RAPIDS_AVAILABLE = False

try:
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


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
    "late_hour_risk",
    "mismatch_x_risk",
    "distance_x_velocity",
    "high_amount_flag",
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

    late_hour = np.maximum(hour - 21, 0)
    fraud_logit = (
        -6.8
        + 0.025 * tx_amount
        + 3.6 * terminal_risk
        + 1.3 * country_mismatch
        + 0.11 * late_hour
        + 0.09 * velocity_1h
        + 0.006 * distance_from_home
        + 2.0 * (terminal_risk * country_mismatch)
        + 0.013 * (tx_amount * terminal_risk)
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-(1.9 * fraud_logit)))
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
        # cuDF API varies by version; support both conversion entry points.
        if hasattr(cudf.DataFrame, "from_pandas"):
            gdf = cudf.DataFrame.from_pandas(df)
        elif hasattr(cudf, "from_pandas"):
            gdf = cudf.from_pandas(df)
        else:
            gdf = cudf.DataFrame(df)
        gdf["amount_log"] = np.log1p(gdf["tx_amount"])
        gdf["night_tx"] = ((gdf["hour"] <= 5) | (gdf["hour"] >= 22)).astype("int8")
        gdf["amount_x_risk"] = gdf["tx_amount"] * gdf["terminal_risk"]
        gdf["late_hour_risk"] = ((gdf["hour"] - 21) * (gdf["hour"] > 21)).astype("float32")
        gdf["mismatch_x_risk"] = gdf["country_mismatch"] * gdf["terminal_risk"]
        gdf["distance_x_velocity"] = gdf["distance_from_home"] * gdf["velocity_1h"]
        gdf["high_amount_flag"] = (gdf["tx_amount"] >= 120.0).astype("int8")
        return gdf

    out = df.copy()
    out["amount_log"] = np.log1p(out["tx_amount"])
    out["night_tx"] = ((out["hour"] <= 5) | (out["hour"] >= 22)).astype("int8")
    out["amount_x_risk"] = out["tx_amount"] * out["terminal_risk"]
    out["late_hour_risk"] = ((out["hour"] - 21).clip(lower=0)).astype(np.float32)
    out["mismatch_x_risk"] = out["country_mismatch"] * out["terminal_risk"]
    out["distance_x_velocity"] = out["distance_from_home"] * out["velocity_1h"]
    out["high_amount_flag"] = (out["tx_amount"] >= 120.0).astype("int8")
    return out


def to_numpy_features_labels(df_engineered):
    if RAPIDS_AVAILABLE and hasattr(df_engineered, "to_pandas"):
        pdf = df_engineered.to_pandas()
    else:
        pdf = df_engineered

    x = pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = pdf["is_fraud"].to_numpy(dtype=np.int32)
    return x, y, pdf


def _to_numpy(arr):
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    return np.asarray(arr)


def train_and_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    use_gpu: bool,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    positives = max(1, int(y_train.sum()))
    negatives = max(1, int(y_train.shape[0] - positives))
    scale_pos_weight = float(np.sqrt(negatives / positives))

    params = {
        "objective": "binary:logistic",
        # Keep early stopping focused on ranking quality instead of logloss.
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 7,
        "eta": 0.05,
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
    }

    # XGBoost 2.x prefers device selection over legacy gpu_hist.
    params["device"] = "cuda" if use_gpu else "cpu"

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    pred = booster.predict(dtest)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, pred)),
        "test_rows": int(y_test.shape[0]),
        "fraud_rate_test": float(y_test.mean()),
        "best_iteration": int(booster.best_iteration)
        if booster.best_iteration is not None
        else None,
        "scale_pos_weight": float(scale_pos_weight),
    }
    return booster, metrics, x_test, y_test


def run_inference_evaluation(
    booster,
    use_gpu: bool,
    seed: int,
    rows: int,
    threshold: float,
    use_rapids: bool,
):
    raw_infer = generate_synthetic_transactions(rows, seed + 1)
    engineered_infer = feature_engineer(raw_infer, use_rapids=use_rapids)

    if use_gpu and RAPIDS_AVAILABLE and hasattr(engineered_infer, "__getitem__"):
        infer_features = engineered_infer[FEATURE_COLUMNS].astype("float32")
    else:
        infer_pdf = (
            engineered_infer.to_pandas()
            if hasattr(engineered_infer, "to_pandas")
            else engineered_infer
        )
        infer_features = infer_pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    infer_truth = _to_numpy(
        engineered_infer["is_fraud"]
        if hasattr(engineered_infer, "__getitem__")
        else raw_infer["is_fraud"]
    ).astype(np.int32)

    infer_start = time.perf_counter()
    if use_gpu:
        pred_scores_obj = booster.inplace_predict(infer_features)
    else:
        dinfer = xgb.DMatrix(infer_features, feature_names=FEATURE_COLUMNS)
        pred_scores_obj = booster.predict(dinfer)
    infer_seconds = time.perf_counter() - infer_start

    pred_scores = _to_numpy(pred_scores_obj).astype(np.float32)
    pred_label = (pred_scores >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        infer_truth,
        pred_label,
        average="binary",
        zero_division=0,
    )

    report = {
        "rows": int(rows),
        "threshold": float(threshold),
        "inference_seconds": float(infer_seconds),
        "rows_per_second": float(rows / max(infer_seconds, 1e-9)),
        "ms_per_1000_rows": float((infer_seconds * 1000.0) / max(rows / 1000.0, 1e-9)),
        "roc_auc": float(roc_auc_score(infer_truth, pred_scores)),
        "pr_auc": float(average_precision_score(infer_truth, pred_scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fraud_rate": float(infer_truth.mean()),
    }

    pred_frame = pd.DataFrame(
        {
            "is_fraud": infer_truth,
            "fraud_score": pred_scores,
            "predicted_fraud": pred_label,
        }
    )
    return report, pred_frame


def persist_artifacts(
    out_dir: Path,
    booster,
    train_metrics: dict,
    inference_metrics: dict,
    engineered_pdf: pd.DataFrame,
    inference_predictions: pd.DataFrame,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "fraud_xgb_model.json"
    train_metrics_path = out_dir / "metrics_train.json"
    inference_metrics_path = out_dir / "metrics_inference.json"
    data_path = out_dir / "engineered_transactions.parquet"
    pred_path = out_dir / "inference_predictions.parquet"
    manifest_path = out_dir / "pipeline_manifest.json"

    booster.save_model(model_path.as_posix())
    train_metrics_path.write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    inference_metrics_path.write_text(json.dumps(inference_metrics, indent=2), encoding="utf-8")
    engineered_pdf.to_parquet(data_path, index=False)
    inference_predictions.to_parquet(pred_path, index=False)

    manifest = {
        "feature_columns": FEATURE_COLUMNS,
        "model_path": model_path.as_posix(),
        "train_metrics_path": train_metrics_path.as_posix(),
        "inference_metrics_path": inference_metrics_path.as_posix(),
        "engineered_data_path": data_path.as_posix(),
        "inference_predictions_path": pred_path.as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return (
        model_path,
        train_metrics_path,
        inference_metrics_path,
        data_path,
        pred_path,
        manifest_path,
    )


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
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=400,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds on validation set",
    )
    parser.add_argument(
        "--inference-rows",
        type=int,
        default=120000,
        help="Rows for post-training inference benchmark",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Fraud score threshold for binary classification",
    )
    args = parser.parse_args()

    use_rapids = RAPIDS_AVAILABLE and not args.force_cpu
    use_gpu_for_training = use_rapids and not args.force_cpu

    raw = generate_synthetic_transactions(args.rows, args.seed)
    engineered = feature_engineer(raw, use_rapids=use_rapids)
    x, y, engineered_pdf = to_numpy_features_labels(engineered)

    booster, train_metrics, _, _ = train_and_evaluate(
        x,
        y,
        use_gpu=use_gpu_for_training,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    inference_metrics, inference_predictions = run_inference_evaluation(
        booster=booster,
        use_gpu=use_gpu_for_training,
        seed=args.seed,
        rows=args.inference_rows,
        threshold=args.decision_threshold,
        use_rapids=use_rapids,
    )

    model_path, train_metrics_path, inference_metrics_path, data_path, pred_path, manifest_path = (
        persist_artifacts(
            args.output_dir,
            booster,
            train_metrics,
            inference_metrics,
            engineered_pdf,
            inference_predictions,
        )
    )

    runtime_mode = "GPU (RAPIDS + CUDA)" if use_gpu_for_training else "CPU fallback"
    print(f"Run mode: {runtime_mode}")
    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("Inference metrics:")
    print(json.dumps(inference_metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved train metrics: {train_metrics_path}")
    print(f"Saved inference metrics: {inference_metrics_path}")
    print(f"Saved engineered data: {data_path}")
    print(f"Saved inference predictions: {pred_path}")
    print(f"Saved pipeline manifest: {manifest_path}")


if __name__ == "__main__":
    main()
