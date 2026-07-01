import os
import warnings
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from fingerprint_utils import smiles_to_fingerprint

warnings.filterwarnings("ignore")

DATA_PATH = "data/toxicity_data.csv"
RESULTS_DIR = "results"
EXPERIMENT_NAME = "OA-DeepFPLearn-ML-Ensemble-Comparison"

os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def get_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    preds = model.predict(X)
    return preds.astype(float)


def build_models():
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    gb = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    hgb = HistGradientBoostingClassifier(random_state=42)

    svm = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=42))
    ])

    knn = Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=3))
    ])

    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            random_state=42
        ))
    ])

    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
            ("lr", lr)
        ],
        voting="soft"
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
            ("svm", svm)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=3
    )

    return {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "ExtraTrees": et,
        "GradientBoosting": gb,
        "AdaBoost": ada,
        "HistGradientBoosting": hgb,
        "SVM": svm,
        "KNN": knn,
        "MLP": mlp,
        "VotingEnsemble": voting,
        "StackingEnsemble": stacking
    }


def evaluate_model(model, X, y):
    n_samples = len(y)
    n_classes = len(np.unique(y))

    if n_samples < 8 or n_classes < 2:
        model.fit(X, y)
        preds = model.predict(X)
        probs = get_probability(model, X)
        eval_mode = "train_only_small_dataset"
    else:
        test_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = get_probability(model, X_test)
        y = y_test
        eval_mode = "holdout"

    return {
        "evaluation_mode": eval_mode,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "mcc": matthews_corrcoef(y, preds) if len(np.unique(y)) > 1 else 0.0,
        "roc_auc": safe_auc(y, probs)
    }


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    required = {"dataset", "smiles", "toxicity"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["smiles", "toxicity"])
    df["toxicity"] = df["toxicity"].astype(int)

    return df


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()
    all_results = []

    for dataset_name, dataset_df in df.groupby("dataset"):
        print(f"\nDataset: {dataset_name}")

        X = np.array([
            smiles_to_fingerprint(smiles)
            for smiles in dataset_df["smiles"]
        ], dtype=np.float32)

        y = dataset_df["toxicity"].values.astype(int)
        models = build_models()

        for model_name, model in models.items():
            run_name = f"{dataset_name}_{model_name}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("model", model_name)
                mlflow.log_param("n_samples", len(y))
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("positive_rate", float(np.mean(y)))

                try:
                    metrics = evaluate_model(model, X, y)
                    mlflow.log_metrics({
                        k: float(v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v)
                    })
                    mlflow.log_param("evaluation_mode", metrics["evaluation_mode"])

                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        **metrics
                    }
                    all_results.append(row)

                    print(
                        f"{model_name}: "
                        f"F1={metrics['f1']:.3f}, "
                        f"AUC={metrics['roc_auc'] if not pd.isna(metrics['roc_auc']) else 'NA'}"
                    )

                except Exception as e:
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))
                    print(f"{model_name} failed: {e}")

    results_df = pd.DataFrame(all_results)

    summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    results_df.to_csv(summary_path, index=False)

    best_rows = []

    for dataset_name, group in results_df.groupby("dataset"):
        scored = group.copy()
        scored["selection_score"] = scored["roc_auc"]
        scored.loc[scored["selection_score"].isna(), "selection_score"] = scored["f1"]

        best = scored.sort_values(
            by=["selection_score", "f1", "mcc"],
            ascending=False
        ).iloc[0]

        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    best_path = os.path.join(RESULTS_DIR, "best_models_by_dataset.csv")
    best_df.to_csv(best_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(best_path)

    print("\nBest models by dataset:")
    print(best_df[["dataset", "model", "selection_score", "f1", "roc_auc", "mcc"]])


if __name__ == "__main__":
    main()
