"""
Student Performance: MLflow Experiment Tracking and Data Drift Analysis
Educational demonstration using synthetic data only.
Do not use the generated predictions for real student assessment.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "assignment_completion_pct",
    "sleep_hours",
]
TARGET = "final_score"
RANDOM_STATE = 42


def make_synthetic_dataset(n_rows: int = 240, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create reproducible demonstration data; no external dataset is required."""
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0.5, 8.0, n_rows)
    attendance = rng.uniform(45, 100, n_rows)
    previous_score = rng.uniform(25, 95, n_rows)
    assignment_completion = rng.uniform(30, 100, n_rows)
    sleep_hours = rng.uniform(4.0, 9.5, n_rows)

    noise = rng.normal(0, 5, n_rows)
    final_score = (
        3.4 * study_hours
        + 0.18 * attendance
        + 0.30 * previous_score
        + 0.12 * assignment_completion
        + 1.2 * sleep_hours
        - 8
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    return pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance_pct": attendance.round(2),
        "previous_score": previous_score.round(2),
        "assignment_completion_pct": assignment_completion.round(2),
        "sleep_hours": sleep_hours.round(2),
        "final_score": final_score.round(2),
    })


def run_model_experiment(reference_df: pd.DataFrame) -> dict:
    """Train a model and log parameters, metrics, and artifacts to MLflow."""
    X = reference_df[FEATURES]
    y = reference_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    scored = X_test.copy()
    scored["actual_score"] = y_test.to_numpy()
    scored["predicted_score"] = predictions.round(2)
    predictions_path = OUTPUT_DIR / "model_predictions.csv"
    scored.to_csv(predictions_path, index=False)

    with mlflow.start_run(run_name="student-performance-baseline") as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("dataset_type", "synthetic_demo")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        # Log the fitted model in the current MLflow sklearn flavor.
        try:
            mlflow.sklearn.log_model(model, name="student_performance_model")
        except (AttributeError, TypeError):
            # Compatibility fallback for older MLflow releases.
            mlflow.sklearn.log_model(model, artifact_path="student_performance_model")
        run_id = run.info.run_id

    return {"model": model, "metrics": metrics, "run_id": run_id}


def run_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Compare numeric feature distributions with two-sample KS tests."""
    rows = []
    for feature in FEATURES:
        statistic, p_value = ks_2samp(
            reference_df[feature].to_numpy(),
            current_df[feature].to_numpy(),
            alternative="two-sided",
            method="auto",
        )
        rows.append({
            "feature": feature,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05),
            "significance_threshold": 0.05,
        })

    results = pd.DataFrame(rows)
    results_path = OUTPUT_DIR / "drift_results.csv"
    results.to_csv(results_path, index=False)

    plt.figure(figsize=(9, 4.8))
    plt.bar(results["feature"], results["ks_statistic"])
    plt.axhline(0.05, linestyle="--", label="p-value threshold is 0.05 (not KS statistic)")
    plt.title("KS Statistic by Feature: Reference vs Current Data")
    plt.ylabel("KS statistic")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "drift_plot.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # JSON summary is convenient for quick review and reproducibility.
    summary_path = OUTPUT_DIR / "drift_summary.json"
    summary_path.write_text(
        json.dumps({
            "test": "two-sample Kolmogorov-Smirnov",
            "threshold": 0.05,
            "drifted_features": results.loc[results["drift_detected"], "feature"].tolist(),
            "note": "A drift flag indicates a distribution difference, not proof of model degradation."
        }, indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="student-performance-data-drift"):
        mlflow.log_param("comparison", "reference_vs_simulated_current")
        mlflow.log_param("drift_test", "two-sample_KS")
        mlflow.log_param("p_value_threshold", 0.05)
        mlflow.log_param("reference_rows", len(reference_df))
        mlflow.log_param("current_rows", len(current_df))
        mlflow.log_metric("features_tested", len(results))
        mlflow.log_metric("features_flagged", int(results["drift_detected"].sum()))
        mlflow.log_artifact(str(results_path), artifact_path="drift")
        mlflow.log_artifact(str(plot_path), artifact_path="drift")
        mlflow.log_artifact(str(summary_path), artifact_path="drift")

    return results


def main() -> None:
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Student_Performance_Data_Drift_Analysis")

    reference = make_synthetic_dataset(n_rows=240, seed=RANDOM_STATE)
    # Create a controlled distribution shift in study_hours for demonstration.
    current = make_synthetic_dataset(n_rows=240, seed=2026)
    current["study_hours"] = np.clip(current["study_hours"] + 2.0, 0.5, 8.0).round(2)
    current["attendance_pct"] = np.clip(current["attendance_pct"] - 5.0, 0, 100).round(2)

    reference.to_csv(OUTPUT_DIR / "reference_data.csv", index=False)
    current.to_csv(OUTPUT_DIR / "simulated_current_data.csv", index=False)

    experiment = run_model_experiment(reference)
    drift_results = run_drift_analysis(reference, current)

    print("\n=== Baseline model metrics ===")
    for key, value in experiment["metrics"].items():
        print(f"{key.upper()}: {value:.4f}")
    print(f"MLflow baseline run ID: {experiment['run_id']}")

    print("\n=== Data drift results ===")
    print(drift_results.to_string(index=False))
    print("\nArtifacts written to:", OUTPUT_DIR)
    print("Start the MLflow UI with: python -m mlflow ui --backend-store-uri ./mlruns --port 5001")
    print("Then open http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
