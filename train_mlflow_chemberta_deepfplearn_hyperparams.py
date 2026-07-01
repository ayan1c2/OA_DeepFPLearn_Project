import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from rdkit import Chem
from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.warning")
from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from ontology_utils import build_ontology_graph, train_ontology_embedding, get_ontology_vector, export_missing_ontology_nodes
from logger_utils import logger

warnings.filterwarnings("ignore")

DATA_PATH = "data/toxicity_data.csv"
MODEL_DIR = "artifacts"

MODEL_PATH = os.path.join(MODEL_DIR, "chemberta_oa_deepfplearn.keras")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_autoencoder.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_encoder.keras")

EXPERIMENT_NAME = "ChemBERTa-OA-DeepFPLearn-PFAS"

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

FINGERPRINT_BITS = 2048
ONTOLOGY_DIM = 32
LATENT_FP_DIM = 64
CHEMBERTA_DIM = 768
MAX_SMILES_LENGTH = 128
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)

HYPERPARAMS = {
    "latent_fingerprint_dim": 128,
    "autoencoder_learning_rate": 0.0005,
    "classifier_learning_rate": 0.00025,
    "autoencoder_epochs": 120,
    "classifier_epochs": 140,
    "batch_size": 16,
    "encoder_dropout": 0.25,
    "chemberta_dropout": 0.25,
    "ontology_dropout": 0.20,
    "classifier_dropout": 0.35,
    "l2_regularization": 1e-5,
    "early_stopping_patience": 18,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_learning_rate": 1e-6,
    "decision_threshold": 0.45,
    "chemberta_batch_size": 16
}



def safe_mol_from_smiles(smiles):
    try:
        if smiles is None or pd.isna(smiles):
            return None
        smiles = str(smiles).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            return None
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning("Invalid SMILES skipped: %s", smiles)
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception as exc:
        logger.warning("Problematic SMILES skipped: %s | %s", smiles, exc)
        return None

def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS):
    mol = safe_mol_from_smiles(smiles)

    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    generator = MORGAN_GENERATOR if n_bits == FINGERPRINT_BITS else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def load_chemberta():
    """
    Load ChemBERTa with PyTorch instead of TFAutoModel.

    Reason: recent Transformers TensorFlow models can fail with Keras 3 unless
    the separate tf-keras compatibility package is installed. Since ChemBERTa is
    used only to precompute fixed SMILES embeddings, PyTorch AutoModel avoids
    the Keras 3 / TFAutoModel compatibility problem while keeping the downstream
    OA-DeepFPLearn classifier in TensorFlow/Keras.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
        chemberta = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME)
        chemberta.eval()
        return tokenizer, chemberta
    except Exception as exc:
        raise RuntimeError(
            "ChemBERTa loading failed. Install PyTorch support with: "
            "pip install torch transformers"
            f" | Original error: {exc}"
        ) from exc


def smiles_to_chemberta_embeddings(smiles_list, tokenizer, chemberta, batch_size=HYPERPARAMS["chemberta_batch_size"]):
    all_embeddings = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chemberta.to(device)

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = [str(x) for x in smiles_list[start:start + batch_size]]

            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=MAX_SMILES_LENGTH,
                return_tensors="pt"
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = chemberta(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.detach().cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def build_fingerprint_autoencoder(fingerprint_dim, latent_dim=HYPERPARAMS["latent_fingerprint_dim"]):
    fp_input = tf.keras.Input(
        shape=(fingerprint_dim,),
        name="fingerprint_autoencoder_input"
    )

    x = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    x = tf.keras.layers.Dropout(HYPERPARAMS["encoder_dropout"])(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="relu",
        name="latent_fingerprint_vector"
    )(x)

    x = tf.keras.layers.Dense(256, activation="relu")(latent)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)

    reconstructed = tf.keras.layers.Dense(
        fingerprint_dim,
        activation="sigmoid",
        name="reconstructed_fingerprint"
    )(x)

    autoencoder = tf.keras.Model(
        fp_input,
        reconstructed,
        name="DeepFPLearn_Fingerprint_Autoencoder"
    )

    encoder = tf.keras.Model(
        fp_input,
        latent,
        name="DeepFPLearn_Fingerprint_Encoder"
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS["autoencoder_learning_rate"]),
        loss="binary_crossentropy",
        metrics=["mse"]
    )

    return autoencoder, encoder


def build_chemberta_oa_deepfplearn_model(
    fingerprint_dim,
    chemberta_dim,
    ontology_dim,
    latent_dim=HYPERPARAMS["latent_fingerprint_dim"]
):
    fp_input = tf.keras.Input(
        shape=(fingerprint_dim,),
        name="fingerprint_input"
    )

    chemberta_input = tf.keras.Input(
        shape=(chemberta_dim,),
        name="chemberta_input"
    )

    ontology_input = tf.keras.Input(
        shape=(ontology_dim,),
        name="ontology_input"
    )

    fp = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    fp = tf.keras.layers.Dropout(HYPERPARAMS["encoder_dropout"])(fp)
    fp = tf.keras.layers.Dense(256, activation="relu")(fp)

    latent_fp = tf.keras.layers.Dense(
        latent_dim,
        activation="relu",
        name="deepfplearn_latent_fingerprint"
    )(fp)

    chem = tf.keras.layers.Dense(256, activation="relu")(chemberta_input)
    chem = tf.keras.layers.Dropout(HYPERPARAMS["chemberta_dropout"])(chem)
    chem = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="chemberta_smiles_embedding"
    )(chem)

    onto = tf.keras.layers.Dense(128, activation="relu")(ontology_input)
    onto = tf.keras.layers.Dropout(HYPERPARAMS["ontology_dropout"])(onto)
    onto = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="ontology_semantic_embedding"
    )(onto)

    fused = tf.keras.layers.Concatenate(name="fp_chemberta_ontology_fusion")(
        [latent_fp, chem, onto]
    )

    attention_weights = tf.keras.layers.Dense(
        fused.shape[-1],
        activation="sigmoid",
        name="attention_weights"
    )(fused)

    attended = tf.keras.layers.Multiply(name="attention_weighted_fusion")(
        [fused, attention_weights]
    )

    x = tf.keras.layers.Dense(128, activation="relu")(attended)
    x = tf.keras.layers.Dropout(HYPERPARAMS["classifier_dropout"])(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="toxicity_probability"
    )(x)

    model = tf.keras.Model(
        inputs=[fp_input, chemberta_input, ontology_input],
        outputs=output,
        name="ChemBERTa_OA_DeepFPLearn"
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
    df = df.dropna(
        subset=["smiles", "pfas_class", "toxicity"]
    ).reset_index(drop=True)

    graph = build_ontology_graph()
    ontology_embedding_model = train_ontology_embedding(
        graph,
        dimensions=ONTOLOGY_DIM
    )

    tokenizer, chemberta = load_chemberta()

    X_fp = np.array(
        [smiles_to_fingerprint(smiles) for smiles in df["smiles"]],
        dtype=np.float32
    )

    X_chemberta = smiles_to_chemberta_embeddings(
        df["smiles"].astype(str).tolist(),
        tokenizer,
        chemberta
    )

    X_onto = np.array(
        [
            get_ontology_vector(
                class_name,
                ontology_embedding_model,
                dimensions=ONTOLOGY_DIM
            )
            for class_name in df["pfas_class"]
        ],
        dtype=np.float32
    )

    y = df["toxicity"].values.astype(np.float32)

    return df, X_fp, X_chemberta, X_onto, y, graph


def generate_explainability_artifacts(
    model,
    Xfp_test,
    Xchem_test,
    Xonto_test,
    df_test
):
    explain_dir = os.path.join(MODEL_DIR, "explainability")
    os.makedirs(explain_dir, exist_ok=True)

    attention_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("attention_weights").output
    )

    attention_values = attention_model.predict(
        {
            "fingerprint_input": Xfp_test,
            "chemberta_input": Xchem_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    )

    attention_path = os.path.join(
        explain_dir,
        "fusion_attention_weights_fp_chemberta_ontology.csv"
    )

    pd.DataFrame(attention_values).to_csv(attention_path, index=False)

    probs = model.predict(
        {
            "fingerprint_input": Xfp_test,
            "chemberta_input": Xchem_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    ).ravel()

    df_test = df_test.copy()
    df_test["predicted_probability"] = probs
    df_test["predicted_class"] = (probs >= HYPERPARAMS["decision_threshold"]).astype(int)

    prediction_path = os.path.join(
        explain_dir,
        "prediction_explanations.csv"
    )

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

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog(log_models=False)

    df, X_fp, X_chemberta, X_onto, y, graph = prepare_dataset()

    validated_arrays, dataset_stats_path = validate_training_arrays(
        df,
        {"X_fp": X_fp, "X_chemberta": X_chemberta, "X_onto": X_onto, "y": y}
    )
    X_fp = validated_arrays["X_fp"]
    X_chemberta = validated_arrays["X_chemberta"]
    X_onto = validated_arrays["X_onto"]
    y = validated_arrays["y"]

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two toxicity classes: 0 and 1.")

    stratify = y if min(np.bincount(y.astype(int))) >= 2 else None

    (
        Xfp_train,
        Xfp_test,
        Xchem_train,
        Xchem_test,
        Xonto_train,
        Xonto_test,
        y_train,
        y_test,
        df_train,
        df_test
    ) = train_test_split(
        X_fp,
        X_chemberta,
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
        "model_type": "ChemBERTa-OA-DeepFPLearn",
        "description": "ChemBERTa SMILES embeddings + DeepFPLearn latent fingerprints + ontology embeddings + attention fusion",
        "chemberta_model": CHEMBERTA_MODEL_NAME,
        "fingerprint_dim": X_fp.shape[1],
        "chemberta_dim": X_chemberta.shape[1],
        "ontology_dim": X_onto.shape[1],
        "latent_fingerprint_dim": HYPERPARAMS["latent_fingerprint_dim"],
        "autoencoder_epochs": HYPERPARAMS["autoencoder_epochs"],
        "classifier_epochs": HYPERPARAMS["classifier_epochs"],
        "batch_size": HYPERPARAMS["batch_size"],
        "framework": "TensorFlow/Keras",
        "hyperparameter_config": json.dumps(HYPERPARAMS)
    }

    with mlflow.start_run(run_name="chemberta_oa_deepfplearn_run"):
        mlflow.log_params(params)

        autoencoder, encoder = build_fingerprint_autoencoder(
            fingerprint_dim=X_fp.shape[1],
            latent_dim=HYPERPARAMS["latent_fingerprint_dim"]
        )

        autoencoder.fit(
            Xfp_train,
            Xfp_train,
            validation_split=0.2,
            epochs=params["autoencoder_epochs"],
            batch_size=params["batch_size"],
            callbacks=[
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
            ],
            verbose=1
        )

        autoencoder.save(AUTOENCODER_PATH)
        encoder.save(ENCODER_PATH)

        model = build_chemberta_oa_deepfplearn_model(
            fingerprint_dim=X_fp.shape[1],
            chemberta_dim=X_chemberta.shape[1],
            ontology_dim=X_onto.shape[1],
            latent_dim=HYPERPARAMS["latent_fingerprint_dim"]
        )

        model.fit(
            {
                "fingerprint_input": Xfp_train,
                "chemberta_input": Xchem_train,
                "ontology_input": Xonto_train
            },
            y_train,
            validation_split=0.2,
            class_weight=class_weights,
            epochs=params["classifier_epochs"],
            batch_size=params["batch_size"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=10,
                    mode="max",
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        probs = model.predict(
            {
                "fingerprint_input": Xfp_test,
                "chemberta_input": Xchem_test,
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

        ontology_artifact = os.path.join(
            MODEL_DIR,
            "ontology_edges.json"
        )

        with open(ontology_artifact, "w") as f:
            json.dump(list(graph.edges()), f, indent=2)

        metadata_path = os.path.join(
            MODEL_DIR,
            "model_metadata.json"
        )

        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "model_name": "ChemBERTa-OA-DeepFPLearn",
                    "architecture": [
                        "RDKit Morgan fingerprint input",
                        "DeepFPLearn-style fingerprint autoencoder",
                        "Latent fingerprint representation",
                        "ChemBERTa SMILES transformer embedding branch",
                        "Ontology embedding branch",
                        "Attention-based fusion",
                        "Binary PFAS toxicity prediction",
                        "Fusion attention explainability artifacts"
                    ],
                    "metrics": metrics
                },
                f,
                indent=2
            )

        attention_path, prediction_path = generate_explainability_artifacts(
            model,
            Xfp_test,
            Xchem_test,
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
        print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()