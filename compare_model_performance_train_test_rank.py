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
    "ChemBERTa-DeepFPLearn": "ChemBERTa-OA-DeepFPLearn-PFAS",
    "GraphSAGE-DeepFPLearn": "OA-DeepFPLearn-GraphSAGE-PFAS",
    "GraphSAGE++-DeepFPLearn": "OA-DeepFPLearn-GraphSAGEPlus-PFAS"
}

METRICS = [
    "train_accuracy","train_precision","train_recall","train_f1","train_roc_auc",
    "val_accuracy","val_precision","val_recall","val_f1","val_roc_auc",
    "test_accuracy","test_precision","test_recall","test_f1","test_roc_auc",
    "test_pr_auc","test_mcc"
]

def get_best_run_for_experiment(client, experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1
    )
    return runs[0] if runs else None

def collect_results():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    rows = []
    for model, exp_name in EXPERIMENTS.items():
        run = get_best_run_for_experiment(client, exp_name)
        if run is None:
            continue
        row = {"model": model, "experiment": exp_name, "run_id": run.info.run_id}
        for m in METRICS:
            row[m] = run.data.metrics.get(m, None)
        rows.append(row)
    return pd.DataFrame(rows)

def normalize(series):
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def create_ranking(df):
    df = df.copy()
    for m in ["test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"]:
        if m in df.columns:
            df[f"norm_{m}"] = normalize(df[m])

    df["overfit_score"] = (
        (df["train_accuracy"] - df["test_accuracy"]).abs().fillna(0) +
        (df["train_roc_auc"] - df["test_roc_auc"]).abs().fillna(0)
    )

    df["overall_score"] = (
        0.30 * df.get("norm_test_roc_auc", 0) +
        0.25 * df.get("norm_test_f1", 0) +
        0.15 * df.get("norm_test_precision", 0) +
        0.15 * df.get("norm_test_recall", 0) +
        0.10 * df.get("norm_test_accuracy", 0) +
        0.05 * (1 - normalize(df["overfit_score"]))
    )

    df["rank"] = df["overall_score"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("rank")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = collect_results()
    if df.empty:
        print("No MLflow runs found.")
        return

    df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_comparison.csv"), index=False)

    ranked = create_ranking(df)
    ranked.to_csv(os.path.join(OUTPUT_DIR, "overall_model_ranking.csv"), index=False)

    ranked[["model","rank","overall_score"]].to_csv(
        os.path.join(OUTPUT_DIR, "final_rank_table.csv"), index=False
    )

    print(ranked[["model","rank","overall_score"]])

if __name__ == "__main__":
    main()
