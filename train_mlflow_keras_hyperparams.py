import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from fingerprint_utils import smiles_to_fingerprint
from ontology_utils import build_ontology_graph, train_ontology_embedding, get_ontology_vector, export_missing_ontology_nodes
from logger_utils import logger

warnings.filterwarnings("ignore")

DATA_PATH = "data/toxicity_data.csv"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "oa_deepfplearn.keras")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_autoencoder.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_encoder.keras")

EXPERIMENT_NAME = "OA-DeepFPLearn-Keras-PFAS"

HYPERPARAMS = {
    "latent_fingerprint_dim": 128,
    "autoencoder_learning_rate": 0.0005,
    "classifier_learning_rate": 0.0003,
    "autoencoder_epochs": 120,
    "classifier_epochs": 120,
    "batch_size": 16,
    "encoder_dropout": 0.25,
    "ontology_dropout": 0.20,
    "classifier_dropout": 0.35,
    "l2_regularization": 1e-5,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 6,
    "reduce_lr_factor": 0.5,
    "min_learning_rate": 1e-6,
    "decision_threshold": 0.45
}


def build_fingerprint_autoencoder(fingerprint_dim, latent_dim=HYPERPARAMS["latent_fingerprint_dim"]):
    fp_input = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_autoencoder_input")

    x = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    x = tf.keras.layers.Dropout(HYPERPARAMS["encoder_dropout"])(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent_fingerprint_vector")(x)

    x = tf.keras.layers.Dense(256, activation="relu")(latent)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    reconstructed = tf.keras.layers.Dense(
        fingerprint_dim,
        activation="sigmoid",
        name="reconstructed_fingerprint"
    )(x)

    autoencoder = tf.keras.Model(fp_input, reconstructed, name="DeepFPLearn_Fingerprint_Autoencoder")
    encoder = tf.keras.Model(fp_input, latent, name="DeepFPLearn_Fingerprint_Encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS["autoencoder_learning_rate"]),
        loss="binary_crossentropy",
        metrics=["mse"]
    )

    return autoencoder, encoder


def build_oa_deepfplearn_model(fingerprint_dim, ontology_dim, latent_dim=HYPERPARAMS["latent_fingerprint_dim"]):
    fp_input = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_input")
    onto_input = tf.keras.Input(shape=(ontology_dim,), name="ontology_input")

    encoder_input = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    encoder_input = tf.keras.layers.Dropout(HYPERPARAMS["encoder_dropout"])(encoder_input)
    encoder_input = tf.keras.layers.Dense(256, activation="relu")(encoder_input)
    latent_fp = tf.keras.layers.Dense(
        latent_dim,
        activation="relu",
        name="deepfplearn_latent_fingerprint"
    )(encoder_input)

    onto = tf.keras.layers.Dense(128, activation="relu")(onto_input)
    onto = tf.keras.layers.Dropout(HYPERPARAMS["ontology_dropout"])(onto)
    onto = tf.keras.layers.Dense(64, activation="relu", name="ontology_semantic_embedding")(onto)

    fused = tf.keras.layers.Concatenate(name="fingerprint_ontology_fusion")([latent_fp, onto])

    attention_weights = tf.keras.layers.Dense(
        fused.shape[-1],
        activation="sigmoid",
        name="attention_weights"
    )(fused)

    attended = tf.keras.layers.Multiply(name="attention_weighted_fusion")([fused, attention_weights])

    x = tf.keras.layers.Dense(128, activation="relu")(attended)
    x = tf.keras.layers.Dropout(HYPERPARAMS["classifier_dropout"])(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="toxicity_probability"
    )(x)

    model = tf.keras.Model(
        inputs=[fp_input, onto_input],
        outputs=output,
        name="OA_DeepFPLearn"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS["classifier_learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model


def prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["smiles", "pfas_class", "toxicity"]).reset_index(drop=True)

    graph = build_ontology_graph()
    ontology_embedding_model = train_ontology_embedding(graph, dimensions=32)

    X_fp = np.array(
        [smiles_to_fingerprint(smiles) for smiles in df["smiles"]],
        dtype=np.float32
    )

    X_onto = np.array(
        [
            get_ontology_vector(class_name, ontology_embedding_model, dimensions=32)
            for class_name in df["pfas_class"]
        ],
        dtype=np.float32
    )

    y = df["toxicity"].values.astype(np.float32)

    return df, X_fp, X_onto, y, graph


def transfer_encoder_weights(autoencoder, classifier):
    auto_layers = autoencoder.layers
    clf_layers = classifier.layers

    for i in range(1, 5):
        try:
            clf_layers[i].set_weights(auto_layers[i].get_weights())
        except Exception:
            pass


def generate_explainability_artifacts(model, Xfp_test, Xonto_test, df_test):
    explain_dir = os.path.join(MODEL_DIR, "explainability")
    os.makedirs(explain_dir, exist_ok=True)

    attention_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("attention_weights").output
    )

    attention_values = attention_model.predict(
        {
            "fingerprint_input": Xfp_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    )

    attention_df = pd.DataFrame(attention_values)
    attention_path = os.path.join(explain_dir, "attention_weights.csv")
    attention_df.to_csv(attention_path, index=False)

    df_test = df_test.copy()
    df_test["predicted_probability"] = model.predict(
        {
            "fingerprint_input": Xfp_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    ).ravel()

    prediction_path = os.path.join(explain_dir, "prediction_explanations.csv")
    df_test.to_csv(prediction_path, index=False)

    return attention_path, prediction_path



def validate_training_arrays(df, arrays):
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Dataset size: %s", len(df))
    logger.info("Toxicity class distribution:\n%s", df["toxicity"].value_counts().to_string())

    for name, arr in arrays.items():
        if np.isnan(arr).any():
            logger.warning("NaN values detected in %s. Replacing with zeros.", name)
            arrays[name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("%s shape: %s", name, arrays[name].shape)

    stats = {
        "dataset_size": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in df["toxicity"].value_counts().to_dict().items()},
        "feature_shapes": {k: list(v.shape) for k, v in arrays.items()}
    }

    stats_path = os.path.join(MODEL_DIR, "dataset_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return arrays, stats_path


def compute_balanced_class_weights(y_train):
    """Return class weights for imbalanced toxicity labels."""
    y_int = np.asarray(y_train).astype(int)
    classes = np.unique(y_int)
    if len(classes) < 2:
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_int
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog(log_models=False)

    df, X_fp, X_onto, y, graph = prepare_dataset()

    validated_arrays, dataset_stats_path = validate_training_arrays(
        df,
        {"X_fp": X_fp, "X_onto": X_onto, "y": y}
    )
    X_fp = validated_arrays["X_fp"]
    X_onto = validated_arrays["X_onto"]
    y = validated_arrays["y"]

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two toxicity classes: 0 and 1.")

    stratify = y if min(np.bincount(y.astype(int))) >= 2 else None

    (
        Xfp_train,
        Xfp_test,
        Xonto_train,
        Xonto_test,
        y_train,
        y_test,
        df_train,
        df_test
    ) = train_test_split(
        X_fp,
        X_onto,
        y,
        df,
        test_size=0.25,
        random_state=42,
        stratify=stratify
    )

    class_weights = compute_balanced_class_weights(y_train)
    logger.info("Class weights: %s", class_weights)

    params = {
        "model_type": "OA-DeepFPLearn",
        "description": "DeepFPLearn latent fingerprint learning + ontology embeddings + attention fusion + explainability",
        "fingerprint_dim": X_fp.shape[1],
        "ontology_dim": X_onto.shape[1],
        "latent_fingerprint_dim": HYPERPARAMS["latent_fingerprint_dim"],
        "autoencoder_epochs": HYPERPARAMS["autoencoder_epochs"],
        "classifier_epochs": HYPERPARAMS["classifier_epochs"],
        "batch_size": HYPERPARAMS["batch_size"],
        "framework": "TensorFlow/Keras",
        "hyperparameter_config": json.dumps(HYPERPARAMS)
    }

    with mlflow.start_run(run_name="oa_deepfplearn_true_architecture"):
        mlflow.log_params(params)

        autoencoder, encoder = build_fingerprint_autoencoder(
            fingerprint_dim=X_fp.shape[1],
            latent_dim=params["latent_fingerprint_dim"]
        )

        autoencoder_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=HYPERPARAMS["early_stopping_patience"],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=HYPERPARAMS["reduce_lr_factor"],
                patience=HYPERPARAMS["reduce_lr_patience"],
                min_lr=HYPERPARAMS["min_learning_rate"],
                verbose=1
            )
        ]

        autoencoder.fit(
            Xfp_train,
            Xfp_train,
            validation_split=0.2,
            epochs=params["autoencoder_epochs"],
            batch_size=params["batch_size"],
            callbacks=autoencoder_callbacks,
            verbose=1
        )

        autoencoder.save(AUTOENCODER_PATH)
        encoder.save(ENCODER_PATH)

        model = build_oa_deepfplearn_model(
            fingerprint_dim=X_fp.shape[1],
            ontology_dim=X_onto.shape[1],
            latent_dim=params["latent_fingerprint_dim"]
        )

        transfer_encoder_weights(autoencoder, model)

        classifier_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=HYPERPARAMS["early_stopping_patience"],
                mode="max",
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=HYPERPARAMS["reduce_lr_factor"],
                patience=HYPERPARAMS["reduce_lr_patience"],
                min_lr=HYPERPARAMS["min_learning_rate"],
                verbose=1
            )
        ]

        model.fit(
            x={
                "fingerprint_input": Xfp_train,
                "ontology_input": Xonto_train
            },
            y=y_train,
            validation_split=0.2,
            class_weight=class_weights,
            epochs=params["classifier_epochs"],
            batch_size=params["batch_size"],
            callbacks=classifier_callbacks,
            verbose=1
        )

        probs = model.predict(
            {
                "fingerprint_input": Xfp_test,
                "ontology_input": Xonto_test
            },
            verbose=0
        ).ravel()

        preds = (probs >= HYPERPARAMS["decision_threshold"]).astype(int)

        metrics = {
            "test_accuracy": accuracy_score(y_test, preds),
            "test_f1": f1_score(y_test, preds, zero_division=0),
            "test_precision": precision_score(y_test, preds, zero_division=0),
            "test_recall": recall_score(y_test, preds, zero_division=0)
        }

        try:
            metrics["test_roc_auc"] = roc_auc_score(y_test, probs)
        except ValueError:
            metrics["test_roc_auc"] = 0.0

        mlflow.log_metrics(metrics)

        model.save(MODEL_PATH)

        ontology_artifact = os.path.join(MODEL_DIR, "ontology_edges.json")
        with open(ontology_artifact, "w") as f:
            json.dump(list(graph.edges()), f, indent=2)

        metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "model_name": "OA-DeepFPLearn",
                    "architecture": [
                        "RDKit molecular fingerprint input",
                        "DeepFPLearn-style autoencoder latent fingerprint learning",
                        "Ontology embedding branch",
                        "Attention-based fusion layer",
                        "Binary toxicity prediction output",
                        "Attention-weight explainability artifacts"
                    ],
                    "metrics": metrics
                },
                f,
                indent=2
            )

        attention_path, prediction_path = generate_explainability_artifacts(
            model,
            Xfp_test,
            Xonto_test,
            df_test
        )

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(AUTOENCODER_PATH)
        mlflow.log_artifact(ENCODER_PATH)
        mlflow.log_artifact(ontology_artifact)
        missing_ontology_path = export_missing_ontology_nodes()
        mlflow.log_artifact(dataset_stats_path)
        mlflow.log_artifact(missing_ontology_path)
        mlflow.log_artifact(metadata_path)
        mlflow.log_artifact(attention_path)
        mlflow.log_artifact(prediction_path)

        print("\nFinal test metrics:")
        print(metrics)
        print(f"\nSaved OA-DeepFPLearn model to: {MODEL_PATH}")
        print(f"Saved autoencoder to: {AUTOENCODER_PATH}")
        print(f"Saved encoder to: {ENCODER_PATH}")


if __name__ == "__main__":
    main()