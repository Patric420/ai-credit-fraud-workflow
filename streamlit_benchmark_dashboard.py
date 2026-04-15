import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


def load_report_from_path(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def load_report_from_upload(uploaded_file) -> Dict[str, Any]:
    return json.load(uploaded_file)


def load_xgb_pipeline_metrics(base_dir: Path) -> Dict[str, Any]:
    train_path = base_dir / "metrics_train.json"
    infer_path = base_dir / "metrics_inference.json"

    if not train_path.exists() or not infer_path.exists():
        return {"available": False}

    train_metrics = json.loads(train_path.read_text(encoding="utf-8"))
    infer_metrics = json.loads(infer_path.read_text(encoding="utf-8"))
    return {"available": True, "train": train_metrics, "inference": infer_metrics}


def build_timing_table(report: Dict[str, Any]) -> pd.DataFrame:
    cpu_timings = report.get("cpu", {}).get("timings_seconds", {})
    gpu_timings = report.get("gpu", {}).get("timings_seconds", {})

    rows = []
    for cpu_key, cpu_value in cpu_timings.items():
        base_name = cpu_key.replace("cpu_", "")
        gpu_key = f"gpu_{base_name}"
        gpu_value = gpu_timings.get(gpu_key)

        speedup = None
        if gpu_value is not None and gpu_value > 0:
            speedup = cpu_value / gpu_value

        rows.append(
            {
                "step": base_name,
                "cpu_seconds": cpu_value,
                "gpu_seconds": gpu_value,
                "speedup_x": speedup,
            }
        )

    return pd.DataFrame(rows)


def build_metric_table(report: Dict[str, Any]) -> pd.DataFrame:
    cpu_metrics = report.get("cpu", {}).get("metrics", {})
    gpu_metrics = report.get("gpu", {}).get("metrics", {})

    rows = []
    metric_names = sorted(set(cpu_metrics.keys()) | set(gpu_metrics.keys()))
    for name in metric_names:
        rows.append(
            {
                "metric": name,
                "cpu": cpu_metrics.get(name),
                "gpu": gpu_metrics.get(name),
            }
        )

    return pd.DataFrame(rows)


def render_header(report: Dict[str, Any]) -> None:
    st.title("Fraud Pipeline Benchmark Dashboard")
    st.caption("Compare pandas/scikit-learn (CPU) vs cuDF/cuML (GPU) step by step.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{report.get('rows', 0):,}")
    c2.metric("Seed", str(report.get("seed", "-")))
    c3.metric(
        "Data Generation (s)",
        f"{report.get('data_generation_seconds', 0.0):.4f}",
    )


def render_metrics(report: Dict[str, Any]) -> None:
    st.subheader("Model Quality")

    cpu_metrics = report.get("cpu", {}).get("metrics", {})
    gpu_metrics = report.get("gpu", {}).get("metrics", {})
    gpu_available = report.get("gpu", {}).get("available", False)

    m1, m2, m3 = st.columns(3)
    m1.metric("CPU ROC AUC", f"{cpu_metrics.get('roc_auc', 0.0):.4f}")
    m2.metric("CPU PR AUC", f"{cpu_metrics.get('pr_auc', 0.0):.4f}")
    m3.metric("CPU Test Rows", f"{int(cpu_metrics.get('test_rows', 0)):,}")

    if gpu_available and gpu_metrics:
        g1, g2, g3 = st.columns(3)
        g1.metric("GPU ROC AUC", f"{gpu_metrics.get('roc_auc', 0.0):.4f}")
        g2.metric("GPU PR AUC", f"{gpu_metrics.get('pr_auc', 0.0):.4f}")
        g3.metric("GPU Test Rows", f"{int(gpu_metrics.get('test_rows', 0)):,}")

    metric_table = build_metric_table(report)
    if not metric_table.empty:
        st.dataframe(metric_table, use_container_width=True)


def render_timings(report: Dict[str, Any]) -> None:
    st.subheader("Execution Timings")

    timing_df = build_timing_table(report)
    if timing_df.empty:
        st.warning("No timing data found in report.")
        return

    display_df = timing_df.copy()
    for col in ["cpu_seconds", "gpu_seconds", "speedup_x"]:
        if col in display_df.columns:
            # Mixed None/float columns become object dtype in pandas.
            # Coerce to numeric so rounding works consistently.
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(6)

    st.dataframe(display_df, use_container_width=True)

    chart_cpu = timing_df[["step", "cpu_seconds"]].set_index("step")
    st.write("CPU step timings")
    st.bar_chart(chart_cpu)

    gpu_has_data = timing_df["gpu_seconds"].notna().any()
    if gpu_has_data:
        chart_both = timing_df[["step", "cpu_seconds", "gpu_seconds"]].set_index("step")
        st.write("CPU vs GPU timings")
        st.bar_chart(chart_both)


def render_speedups(report: Dict[str, Any]) -> None:
    st.subheader("Speedup Analysis")

    speedups = report.get("speedups", {})
    if not speedups.get("available", False):
        st.info(speedups.get("reason", "GPU speedup data not available."))
        return

    by_step = speedups.get("by_step", {})
    if not by_step:
        st.info("Speedup section is present but empty.")
        return

    rows = []
    for step, values in by_step.items():
        rows.append(
            {
                "step": step,
                "cpu_seconds": values.get("cpu_seconds"),
                "gpu_seconds": values.get("gpu_seconds"),
                "speedup_x": values.get("speedup_x"),
            }
        )

    speedup_df = pd.DataFrame(rows).sort_values("speedup_x", ascending=False)
    st.dataframe(speedup_df.round(6), use_container_width=True)

    chart = speedup_df[["step", "speedup_x"]].set_index("step")
    st.write("Speedup by step (higher is better)")
    st.bar_chart(chart)


def render_xgb_pipeline_metrics(metrics_bundle: Dict[str, Any]) -> None:
    st.subheader("Tuned XGBoost Pipeline (Latest Run)")
    if not metrics_bundle.get("available", False):
        st.info("No tuned XGBoost metrics found in artifacts/metrics_train.json and metrics_inference.json.")
        return

    train_metrics = metrics_bundle.get("train", {})
    infer_metrics = metrics_bundle.get("inference", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train ROC AUC", f"{train_metrics.get('roc_auc', 0.0):.4f}")
    c2.metric("Train PR AUC", f"{train_metrics.get('pr_auc', 0.0):.4f}")
    c3.metric("Infer ROC AUC", f"{infer_metrics.get('roc_auc', 0.0):.4f}")
    c4.metric("Infer PR AUC", f"{infer_metrics.get('pr_auc', 0.0):.4f}")

    table = pd.DataFrame(
        [
            {"phase": "train", **train_metrics},
            {"phase": "inference", **infer_metrics},
        ]
    )
    st.dataframe(table, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Fraud CPU/GPU Benchmark", layout="wide")

    st.sidebar.header("Report Source")
    default_path = "artifacts/cpu_gpu_benchmark_report.json"
    report_path = st.sidebar.text_input("Report path", value=default_path)
    uploaded = st.sidebar.file_uploader("Or upload JSON report", type=["json"])

    try:
        if uploaded is not None:
            report = load_report_from_upload(uploaded)
            st.sidebar.success("Loaded report from uploaded file.")
        else:
            report = load_report_from_path(report_path)
            st.sidebar.success(f"Loaded report from: {report_path}")
    except Exception as exc:
        st.error(f"Could not load benchmark report: {exc}")
        st.stop()

    xgb_metrics = load_xgb_pipeline_metrics(Path("artifacts"))
    render_header(report)
    render_metrics(report)
    render_xgb_pipeline_metrics(xgb_metrics)
    render_timings(report)
    render_speedups(report)

    with st.expander("Raw JSON"):
        st.json(report)


if __name__ == "__main__":
    main()
