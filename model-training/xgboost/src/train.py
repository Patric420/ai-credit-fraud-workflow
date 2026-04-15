import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import cudf
except ImportError:
    cudf = None


LABEL_COLUMN = "TX_FRAUD_1"
MODEL_DIR = Path("/opt/ml/model")


def _read_parquet(path: str, use_gpu: bool):
    if use_gpu:
        if cudf is None:
            raise RuntimeError(
                "USE_GPU=true but RAPIDS cuDF is not installed. Install cudf-cu12 first."
            )
        return cudf.read_parquet(path)
    return pd.read_parquet(path)


def _numeric_feature_columns(df) -> List[str]:
    columns: List[str] = []
    for col in df.columns:
        if col == LABEL_COLUMN:
            continue
        dtype = df[col].dtype
        kind = getattr(dtype, "kind", None)
        if kind in ("i", "u", "f", "b"):
            columns.append(col)
    return columns


def _prepare_xy(df, use_gpu: bool) -> Tuple[object, object, List[str]]:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    feature_cols = _numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    if use_gpu:
        x = df[feature_cols].astype("float32")
        y = df[LABEL_COLUMN].astype("int32")
    else:
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[LABEL_COLUMN].to_numpy(dtype=np.int32)

    return x, y, feature_cols


def train_xgboost(train_data_path: str, test_data_path: str, boosting_rounds: int, use_gpu: bool):
    train_df = _read_parquet(train_data_path, use_gpu=use_gpu)
    test_df = _read_parquet(test_data_path, use_gpu=use_gpu)

    x_train, y_train, feature_cols = _prepare_xy(train_df, use_gpu=use_gpu)
    x_test, y_test, _ = _prepare_xy(test_df, use_gpu=use_gpu)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 8,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
    }

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_cols)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=boosting_rounds,
        evals=[(dtrain, "train"), (dtest, "valid")],
        verbose_eval=False,
    )

    pred = booster.predict(dtest)
    if use_gpu and hasattr(y_test, "to_numpy"):
        y_test_np = y_test.to_numpy()
    else:
        y_test_np = y_test

    metrics = {
        "roc_auc": float(roc_auc_score(y_test_np, pred)),
        "pr_auc": float(average_precision_score(y_test_np, pred)),
        "test_rows": int(len(y_test_np)),
        "fraud_rate_test": float(np.mean(y_test_np)),
        "mode": "gpu" if use_gpu else "cpu",
    }
    return booster, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boost_round", type=int, default=100)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    args = parser.parse_args()

    use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
    booster, metrics = train_xgboost(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        boosting_rounds=args.boost_round,
        use_gpu=use_gpu,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model((MODEL_DIR / "model.xgb").as_posix())
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
