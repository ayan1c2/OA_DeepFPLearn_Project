import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
OUTPUT_DIR = "comparison_results"

EXPERIMENTS = {
    "Encoder-DeepFPLearn": "OA-DeepFPLearn-Keras-PFAS",
    "GNN-DeepFPLearn": "OA-DeepFPLearn-GNN-PFAS",
    "GAT-DeepFPLearn": "OA-DeepFPLearn-GAT-PFAS",
    "ChemBERTa-DeepFPLearn": "ChemBERTa-OA-DeepFPLearn-PFAS"
}

METRICS = [
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc"
]


def get_best_run_for_experiment(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment not found: {experiment_name}")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1
    )

    if not runs:
        print(f"No runs found for experiment: {experiment_name}")
        return None

    return runs[0]


def collect_results():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    results = []

    for model_name, experiment_name in EXPERIMENTS.items():
        run = get_best_run_for_experiment(client, experiment_name)

        if run is None:
            continue

        row = {
            "model": model_name,
            "experiment": experiment_name,
            "run_id": run.info.run_id
        }

        for metric in METRICS:
            row[metric] = run.data.metrics.get(metric, None)

        results.append(row)

    return pd.DataFrame(results)


def save_comparison_table(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "model_performance_comparison.csv")
    df.to_csv(output_path, index=False)

    print("\nModel Performance Comparison:")
    print(df)

    print(f"\nSaved comparison table to: {output_path}")


def plot_metric_comparison(df):
    for metric in METRICS:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(10, 6))
        plt.bar(df["model"], df[metric])
        plt.title(f"Model Comparison: {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f"{metric}_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved plot: {plot_path}")


def save_best_model_summary(df):
    best_rows = []

    for metric in METRICS:
        if metric in df.columns and df[metric].notna().any():
            best_idx = df[metric].idxmax()
            best_rows.append({
                "metric": metric,
                "best_model": df.loc[best_idx, "model"],
                "best_score": df.loc[best_idx, metric],
                "run_id": df.loc[best_idx, "run_id"]
            })

    best_df = pd.DataFrame(best_rows)

    best_path = os.path.join(OUTPUT_DIR, "best_model_by_metric.csv")
    best_df.to_csv(best_path, index=False)

    print("\nBest Model by Metric:")
    print(best_df)

    print(f"\nSaved best-model summary to: {best_path}")


def main():
    df = collect_results()

    if df.empty:
        print("No MLflow runs found. Train the models first.")
        return

    save_comparison_table(df)
    plot_metric_comparison(df)
    save_best_model_summary(df)


if __name__ == "__main__":
    main()