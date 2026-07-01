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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from ontology_utils import build_ontology_graph, train_ontology_embedding, get_ontology_vector

warnings.filterwarnings("ignore")

DATA_PATH = "data/toxicity_data.csv"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "oa_deepfplearn_gat.keras")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_autoencoder.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "fingerprint_encoder.keras")

EXPERIMENT_NAME = "OA-DeepFPLearn-GAT-PFAS"

MAX_ATOMS = 80
ATOM_FEATURE_DIM = 10
FINGERPRINT_BITS = 2048
ONTOLOGY_DIM = 32
LATENT_FP_DIM = 64
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)


def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetNumRadicalElectrons(),
        int(atom.IsInRing()),
        atom.GetMass() / 200.0
    ], dtype=np.float32)


def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = Chem.MolFromSmiles(smiles)

    X = np.zeros((max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)

    if mol is None:
        return X, A

    for i, atom in enumerate(list(mol.GetAtoms())[:max_atoms]):
        X[i] = atom_features(atom)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if i < max_atoms and j < max_atoms:
            A[i, j] = 1.0
            A[j, i] = 1.0

    A += np.eye(max_atoms, dtype=np.float32)

    return X, A


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    generator = MORGAN_GENERATOR if n_bits == FINGERPRINT_BITS else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


@tf.keras.utils.register_keras_serializable(package="OADeepFPLearn")
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        # IMPORTANT: create Dropout once here, not inside call().
        # Creating layers inside call() creates new tf.Variables during tf.function tracing.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]

        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_weight"
        )

        self.a_src = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_attention_source"
        )

        self.a_dst = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_attention_target"
        )

        super().build(input_shape)

    def call(self, inputs, training=False):
        X, A = inputs

        H = tf.matmul(X, self.W)

        src_scores = tf.matmul(H, self.a_src)
        dst_scores = tf.matmul(H, self.a_dst)

        attention_logits = src_scores + tf.transpose(dst_scores, perm=[0, 2, 1])
        attention_logits = tf.nn.leaky_relu(attention_logits, alpha=0.2)

        mask = tf.where(A > 0, 0.0, -1e9)
        attention_logits = attention_logits + mask

        attention = tf.nn.softmax(attention_logits, axis=-1)
        attention = self.dropout(attention, training=training)

        output = tf.matmul(attention, H)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config


def build_fingerprint_autoencoder(fingerprint_dim, latent_dim=LATENT_FP_DIM):
    fp_input = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_autoencoder_input")

    x = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    x = tf.keras.layers.Dropout(0.2)(x)
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["mse"]
    )

    return autoencoder, encoder


def build_oa_deepfplearn_gat_model(
    fingerprint_dim,
    ontology_dim,
    atom_feature_dim=ATOM_FEATURE_DIM,
    max_atoms=MAX_ATOMS,
    latent_dim=LATENT_FP_DIM
):
    fp_input = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_input")
    atom_input = tf.keras.Input(shape=(max_atoms, atom_feature_dim), name="atom_features_input")
    adjacency_input = tf.keras.Input(shape=(max_atoms, max_atoms), name="adjacency_input")
    onto_input = tf.keras.Input(shape=(ontology_dim,), name="ontology_input")

    fp = tf.keras.layers.Dense(1024, activation="relu")(fp_input)
    fp = tf.keras.layers.Dropout(0.2)(fp)
    fp = tf.keras.layers.Dense(256, activation="relu")(fp)

    latent_fp = tf.keras.layers.Dense(
        latent_dim,
        activation="relu",
        name="deepfplearn_latent_fingerprint"
    )(fp)

    gat = GraphAttentionLayer(64, dropout_rate=0.2, name="gat_layer_1")(
        [atom_input, adjacency_input]
    )
    gat = GraphAttentionLayer(128, dropout_rate=0.2, name="gat_layer_2")(
        [gat, adjacency_input]
    )
    gat = GraphAttentionLayer(128, dropout_rate=0.2, name="gat_layer_3")(
        [gat, adjacency_input]
    )

    gat_embedding = tf.keras.layers.GlobalAveragePooling1D(
        name="gat_global_pooling"
    )(gat)

    gat_embedding = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="gat_molecular_embedding"
    )(gat_embedding)

    onto = tf.keras.layers.Dense(128, activation="relu")(onto_input)
    onto = tf.keras.layers.Dropout(0.2)(onto)
    onto = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="ontology_semantic_embedding"
    )(onto)

    fused = tf.keras.layers.Concatenate(name="fp_gat_ontology_fusion")(
        [latent_fp, gat_embedding, onto]
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
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="toxicity_probability"
    )(x)

    model = tf.keras.Model(
        inputs=[fp_input, atom_input, adjacency_input, onto_input],
        outputs=output,
        name="OA_DeepFPLearn_GAT"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["smiles", "pfas_class", "toxicity"]).reset_index(drop=True)

    graph = build_ontology_graph()
    ontology_embedding_model = train_ontology_embedding(graph, dimensions=ONTOLOGY_DIM)

    fingerprints = []
    atom_features_list = []
    adjacency_list = []
    ontology_vectors = []

    for _, row in df.iterrows():
        smiles = row["smiles"]
        pfas_class = row["pfas_class"]

        fingerprints.append(smiles_to_fingerprint(smiles))

        atom_x, adj = smiles_to_graph(smiles)
        atom_features_list.append(atom_x)
        adjacency_list.append(adj)

        ontology_vectors.append(
            get_ontology_vector(
                pfas_class,
                ontology_embedding_model,
                dimensions=ONTOLOGY_DIM
            )
        )

    X_fp = np.array(fingerprints, dtype=np.float32)
    X_atoms = np.array(atom_features_list, dtype=np.float32)
    X_adj = np.array(adjacency_list, dtype=np.float32)
    X_onto = np.array(ontology_vectors, dtype=np.float32)
    y = df["toxicity"].values.astype(np.float32)

    return df, X_fp, X_atoms, X_adj, X_onto, y, graph


def generate_explainability_artifacts(model, Xfp_test, Xatoms_test, Xadj_test, Xonto_test, df_test):
    explain_dir = os.path.join(MODEL_DIR, "explainability")
    os.makedirs(explain_dir, exist_ok=True)

    fusion_attention_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("attention_weights").output
    )

    fusion_attention_values = fusion_attention_model.predict(
        {
            "fingerprint_input": Xfp_test,
            "atom_features_input": Xatoms_test,
            "adjacency_input": Xadj_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    )

    fusion_attention_path = os.path.join(
        explain_dir,
        "fusion_attention_weights_fp_gat_ontology.csv"
    )

    pd.DataFrame(fusion_attention_values).to_csv(fusion_attention_path, index=False)

    probs = model.predict(
        {
            "fingerprint_input": Xfp_test,
            "atom_features_input": Xatoms_test,
            "adjacency_input": Xadj_test,
            "ontology_input": Xonto_test
        },
        verbose=0
    ).ravel()

    df_test = df_test.copy()
    df_test["predicted_probability"] = probs
    df_test["predicted_class"] = (probs >= 0.5).astype(int)

    prediction_path = os.path.join(explain_dir, "prediction_explanations.csv")
    df_test.to_csv(prediction_path, index=False)

    return fusion_attention_path, prediction_path


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog(log_models=False)

    df, X_fp, X_atoms, X_adj, X_onto, y, graph = prepare_dataset()

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two toxicity classes: 0 and 1.")

    stratify = y if min(np.bincount(y.astype(int))) >= 2 else None

    (
        Xfp_train,
        Xfp_test,
        Xatoms_train,
        Xatoms_test,
        Xadj_train,
        Xadj_test,
        Xonto_train,
        Xonto_test,
        y_train,
        y_test,
        df_train,
        df_test
    ) = train_test_split(
        X_fp,
        X_atoms,
        X_adj,
        X_onto,
        y,
        df,
        test_size=0.25,
        random_state=42,
        stratify=stratify
    )

    params = {
        "model_type": "OA-DeepFPLearn-GAT",
        "description": "DeepFPLearn latent fingerprints + Graph Attention Network + ontology embeddings + attention fusion",
        "fingerprint_dim": X_fp.shape[1],
        "max_atoms": MAX_ATOMS,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "ontology_dim": X_onto.shape[1],
        "latent_fingerprint_dim": LATENT_FP_DIM,
        "autoencoder_epochs": 50,
        "classifier_epochs": 50,
        "batch_size": 8,
        "framework": "TensorFlow/Keras"
    }

    with mlflow.start_run(run_name="oa_deepfplearn_gat_run"):
        mlflow.log_params(params)

        autoencoder, encoder = build_fingerprint_autoencoder(
            fingerprint_dim=X_fp.shape[1],
            latent_dim=LATENT_FP_DIM
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
                    patience=8,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        autoencoder.save(AUTOENCODER_PATH)
        encoder.save(ENCODER_PATH)

        model = build_oa_deepfplearn_gat_model(
            fingerprint_dim=X_fp.shape[1],
            ontology_dim=X_onto.shape[1]
        )

        model.fit(
            {
                "fingerprint_input": Xfp_train,
                "atom_features_input": Xatoms_train,
                "adjacency_input": Xadj_train,
                "ontology_input": Xonto_train
            },
            y_train,
            validation_split=0.2,
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
                "atom_features_input": Xatoms_test,
                "adjacency_input": Xadj_test,
                "ontology_input": Xonto_test
            },
            verbose=0
        ).ravel()

        preds = (probs >= 0.5).astype(int)

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
                    "model_name": "OA-DeepFPLearn-GAT",
                    "architecture": [
                        "RDKit Morgan fingerprint input",
                        "DeepFPLearn-style autoencoder latent fingerprint learning",
                        "Molecular Graph Attention Network branch",
                        "Ontology embedding branch",
                        "Fusion-level attention mechanism",
                        "Binary toxicity prediction",
                        "Attention-weight explainability artifacts"
                    ],
                    "metrics": metrics
                },
                f,
                indent=2
            )

        fusion_attention_path, prediction_path = generate_explainability_artifacts(
            model,
            Xfp_test,
            Xatoms_test,
            Xadj_test,
            Xonto_test,
            df_test
        )

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(AUTOENCODER_PATH)
        mlflow.log_artifact(ENCODER_PATH)
        mlflow.log_artifact(ontology_artifact)
        mlflow.log_artifact(metadata_path)
        mlflow.log_artifact(fusion_attention_path)
        mlflow.log_artifact(prediction_path)

        print("\nFinal test metrics:")
        print(metrics)
        print(f"\nSaved OA-DeepFPLearn-GAT model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()