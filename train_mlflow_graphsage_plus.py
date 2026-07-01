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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from ontology_utils import build_ontology_graph, train_ontology_embedding, get_ontology_vector

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.warning")

DATA_PATH = "data/toxicity_data.csv"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "oa_deepfplearn_graphsage_plus.keras")
EXPERIMENT_NAME = "OA-DeepFPLearn-GraphSAGEPlus-PFAS"

MAX_ATOMS = 80
ATOM_FEATURE_DIM = 14
FINGERPRINT_BITS = 2048
ONTOLOGY_DIM = 32
LATENT_FP_DIM = 128
BATCH_SIZE = 16
EPOCHS = 120
AUTOENCODER_EPOCHS = 80
LEARNING_RATE = 3e-4
DROPOUT_RATE = 0.30
L2_REG = 1e-5
RANDOM_STATE = 42
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)


def safe_mol_from_smiles(smiles):
    try:
        if smiles is None or pd.isna(smiles):
            return None
        smiles = str(smiles).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            return None
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def atom_features(atom):
    return np.array([
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge(),
        int(atom.GetHybridization()) / 6.0,
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetImplicitValence() / 8.0,
        atom.GetNumRadicalElectrons(),
        int(atom.IsInRing()),
        atom.GetMass() / 250.0,
        int(atom.GetSymbol() == "F"),
        int(atom.GetSymbol() in {"O", "S", "N"}),
        int(atom.GetSymbol() == "C"),
        int(atom.GetSymbol() == "Cl"),
    ], dtype=np.float32)


def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = safe_mol_from_smiles(smiles)
    x = np.zeros((max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)
    if mol is None:
        return x, adj, mask
    atoms = list(mol.GetAtoms())
    for i, atom in enumerate(atoms[:max_atoms]):
        x[i] = atom_features(atom)
        mask[i] = 1.0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            bond_weight = float(bond.GetBondTypeAsDouble())
            adj[i, j] = bond_weight
            adj[j, i] = bond_weight
    adj += np.eye(max_atoms, dtype=np.float32)
    adj *= mask[:, None]
    adj *= mask[None, :]
    return x, adj, mask


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS):
    mol = safe_mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    generator = MORGAN_GENERATOR if n_bits == FINGERPRINT_BITS else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def masked_mean(x, mask):
    mask_exp = tf.expand_dims(mask, -1)
    return tf.reduce_sum(x * mask_exp, axis=1) / tf.maximum(tf.reduce_sum(mask_exp, axis=1), 1.0)


@tf.keras.utils.register_keras_serializable(package="OADeepFPLearn")
class GraphSAGEPlusLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", dropout_rate=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization()
        self.gate_dense = None
        self.message_dense = None
        self.skip_dense = None

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg else None
        self.message_dense = tf.keras.layers.Dense(self.units, activation=None, kernel_regularizer=reg)
        self.skip_dense = tf.keras.layers.Dense(self.units, activation=None, kernel_regularizer=reg)
        self.gate_dense = tf.keras.layers.Dense(self.units, activation="sigmoid", kernel_regularizer=reg)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, adj = inputs
        degree = tf.maximum(tf.reduce_sum(adj, axis=-1, keepdims=True), 1.0)
        neigh_mean = tf.matmul(adj, x) / degree
        neigh_max = tf.reduce_max(tf.where(tf.expand_dims(adj, -1) > 0, tf.expand_dims(x, 1), tf.fill([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1], tf.shape(x)[2]], -1e9)), axis=2)
        neigh_max = tf.where(tf.math.is_finite(neigh_max), neigh_max, tf.zeros_like(neigh_max))
        message = tf.concat([x, neigh_mean, neigh_max], axis=-1)
        h_msg = self.message_dense(message)
        h_skip = self.skip_dense(x)
        gate = self.gate_dense(message)
        h = gate * h_msg + (1.0 - gate) * h_skip
        h = self.norm(h)
        if self.activation is not None:
            h = self.activation(h)
        return self.dropout(h, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation_name, "dropout_rate": self.dropout_rate, "l2_reg": self.l2_reg})
        return config


@tf.keras.utils.register_keras_serializable(package="OADeepFPLearn")
class GraphAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x, mask = inputs
        scores = self.score_dense(x)
        mask_exp = tf.expand_dims(mask, -1)
        scores = tf.where(mask_exp > 0, scores, tf.fill(tf.shape(scores), -1e9))
        weights = tf.nn.softmax(scores, axis=1)
        pooled = tf.reduce_sum(x * weights, axis=1)
        return pooled


def build_autoencoder(fingerprint_dim):
    inp = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_autoencoder_input")
    x = tf.keras.layers.GaussianNoise(0.03)(inp)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    latent = tf.keras.layers.Dense(LATENT_FP_DIM, activation="relu", name="latent_fingerprint_vector")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(latent)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    out = tf.keras.layers.Dense(fingerprint_dim, activation="sigmoid", name="reconstructed_fingerprint")(x)
    autoencoder = tf.keras.Model(inp, out, name="DeepFPLearn_Denoising_Autoencoder")
    encoder = tf.keras.Model(inp, latent, name="DeepFPLearn_Encoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy", metrics=["mse"])
    return autoencoder, encoder


def build_graphsage_plus_model(fingerprint_dim, ontology_dim):
    fp_input = tf.keras.Input(shape=(fingerprint_dim,), name="fingerprint_input")
    atom_input = tf.keras.Input(shape=(MAX_ATOMS, ATOM_FEATURE_DIM), name="atom_features_input")
    adjacency_input = tf.keras.Input(shape=(MAX_ATOMS, MAX_ATOMS), name="adjacency_input")
    atom_mask_input = tf.keras.Input(shape=(MAX_ATOMS,), name="atom_mask_input")
    onto_input = tf.keras.Input(shape=(ontology_dim,), name="ontology_input")

    fp = tf.keras.layers.GaussianNoise(0.02)(fp_input)
    fp = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(fp)
    fp = tf.keras.layers.BatchNormalization()(fp)
    fp = tf.keras.layers.Dropout(DROPOUT_RATE)(fp)
    fp = tf.keras.layers.Dense(512, activation="relu")(fp)
    fp = tf.keras.layers.Dense(256, activation="relu")(fp)
    latent_fp = tf.keras.layers.Dense(LATENT_FP_DIM, activation="relu", name="deepfplearn_latent_fingerprint")(fp)

    h1 = GraphSAGEPlusLayer(96, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG, name="graphsage_plus_layer_1")([atom_input, adjacency_input])
    h2 = GraphSAGEPlusLayer(128, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG, name="graphsage_plus_layer_2")([h1, adjacency_input])
    h3 = GraphSAGEPlusLayer(128, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG, name="graphsage_plus_layer_3")([h2, adjacency_input])
    mean_pool = tf.keras.layers.Lambda(lambda z: masked_mean(z[0], z[1]), name="graphsage_plus_mean_pool")([h3, atom_mask_input])
    attn_pool = GraphAttentionPooling(name="graphsage_plus_attention_pool")([h3, atom_mask_input])
    graph_embedding = tf.keras.layers.Concatenate(name="graphsage_plus_pool_concat")([mean_pool, attn_pool])
    graph_embedding = tf.keras.layers.Dense(160, activation="relu", name="graphsage_plus_molecular_embedding")(graph_embedding)
    graph_embedding = tf.keras.layers.Dropout(DROPOUT_RATE)(graph_embedding)

    onto = tf.keras.layers.Dense(128, activation="relu")(onto_input)
    onto = tf.keras.layers.BatchNormalization()(onto)
    onto = tf.keras.layers.Dropout(DROPOUT_RATE)(onto)
    onto = tf.keras.layers.Dense(64, activation="relu", name="ontology_semantic_embedding")(onto)

    fused = tf.keras.layers.Concatenate(name="fp_graphsage_plus_ontology_fusion")([latent_fp, graph_embedding, onto])
    attention_weights = tf.keras.layers.Dense(fused.shape[-1], activation="sigmoid", name="attention_weights")(fused)
    attended = tf.keras.layers.Multiply(name="attention_weighted_fusion")([fused, attention_weights])
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(attended)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="toxicity_probability")(x)

    model = tf.keras.Model([fp_input, atom_input, adjacency_input, atom_mask_input, onto_input], out, name="OA_DeepFPLearn_GraphSAGEPlus")
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
    return model


def prepare_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["smiles", "pfas_class", "toxicity"]).reset_index(drop=True)
    graph = build_ontology_graph()
    ontology_embedding_model = train_ontology_embedding(graph, dimensions=ONTOLOGY_DIM)
    fps, atoms, adjs, masks, ontos = [], [], [], [], []
    for _, row in df.iterrows():
        fps.append(smiles_to_fingerprint(row["smiles"]))
        atom_x, adj, mask = smiles_to_graph(row["smiles"])
        atoms.append(atom_x)
        adjs.append(adj)
        masks.append(mask)
        ontos.append(get_ontology_vector(row["pfas_class"], ontology_embedding_model, dimensions=ONTOLOGY_DIM))
    return (
        df,
        np.array(fps, dtype=np.float32),
        np.array(atoms, dtype=np.float32),
        np.array(adjs, dtype=np.float32),
        np.array(masks, dtype=np.float32),
        np.array(ontos, dtype=np.float32),
        df["toxicity"].values.astype(np.float32),
        graph,
    )


def clean_arrays(arrays):
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in arrays.items()}


def save_visualizations(model, Xfp_test, Xatoms_test, Xadj_test, Xmasks_test, Xonto_test, df_test, y_test, probs):
    import matplotlib.pyplot as plt
    out_dir = os.path.join(MODEL_DIR, "graphsage_plus_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    attention_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("attention_weights").output)
    attention_values = attention_model.predict({
        "fingerprint_input": Xfp_test,
        "atom_features_input": Xatoms_test,
        "adjacency_input": Xadj_test,
        "atom_mask_input": Xmasks_test,
        "ontology_input": Xonto_test,
    }, verbose=0)
    attention_path = os.path.join(out_dir, "graphsage_plus_fusion_attention_weights.csv")
    pd.DataFrame(attention_values).to_csv(attention_path, index=False)

    pred_df = df_test.copy()
    pred_df["true_toxicity"] = y_test.astype(int)
    pred_df["predicted_probability"] = probs
    pred_df["predicted_class"] = (probs >= 0.5).astype(int)
    pred_path = os.path.join(out_dir, "graphsage_plus_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(probs, bins=20)
    plt.xlabel("Predicted toxicity probability")
    plt.ylabel("Compound count")
    plt.title("GraphSAGE++ Prediction Probability Distribution")
    hist_path = os.path.join(out_dir, "graphsage_plus_probability_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()

    branch_names = ["latent_fingerprint", "graphsage_plus_embedding", "ontology_embedding"]
    splits = [LATENT_FP_DIM, 160, 64]
    branch_means = []
    start = 0
    for size in splits:
        branch_means.append(float(np.mean(attention_values[:, start:start + size])))
        start += size
    plt.figure(figsize=(8, 5))
    plt.bar(branch_names, branch_means)
    plt.ylabel("Mean fusion attention")
    plt.title("GraphSAGE++ Branch Attention")
    branch_path = os.path.join(out_dir, "graphsage_plus_branch_attention.png")
    plt.tight_layout()
    plt.savefig(branch_path, dpi=300)
    plt.close()
    return [attention_path, pred_path, hist_path, branch_path]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog(log_models=False)

    df, X_fp, X_atoms, X_adj, X_masks, X_onto, y, graph = prepare_dataset()
    arrays = clean_arrays({"X_fp": X_fp, "X_atoms": X_atoms, "X_adj": X_adj, "X_masks": X_masks, "X_onto": X_onto, "y": y})
    X_fp, X_atoms, X_adj, X_masks, X_onto, y = arrays["X_fp"], arrays["X_atoms"], arrays["X_adj"], arrays["X_masks"], arrays["X_onto"], arrays["y"]
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two toxicity classes: 0 and 1.")
    stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
    split = train_test_split(X_fp, X_atoms, X_adj, X_masks, X_onto, y, df, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify)
    Xfp_train, Xfp_test, Xatoms_train, Xatoms_test, Xadj_train, Xadj_test, Xmasks_train, Xmasks_test, Xonto_train, Xonto_test, y_train, y_test, df_train, df_test = split

    params = {"model_type": "OA-DeepFPLearn-GraphSAGE++", "fingerprint_dim": int(X_fp.shape[1]), "ontology_dim": int(X_onto.shape[1]), "max_atoms": MAX_ATOMS, "atom_feature_dim": ATOM_FEATURE_DIM, "latent_fingerprint_dim": LATENT_FP_DIM, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "dropout_rate": DROPOUT_RATE, "l2_reg": L2_REG}

    with mlflow.start_run(run_name="oa_deepfplearn_graphsage_plus_run"):
        mlflow.log_params(params)
        autoencoder, encoder = build_autoencoder(X_fp.shape[1])
        autoencoder.fit(Xfp_train, Xfp_train, validation_split=0.2, epochs=AUTOENCODER_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)], verbose=1)
        autoencoder_path = os.path.join(MODEL_DIR, "fingerprint_autoencoder_graphsage_plus.keras")
        encoder_path = os.path.join(MODEL_DIR, "fingerprint_encoder_graphsage_plus.keras")
        autoencoder.save(autoencoder_path)
        encoder.save(encoder_path)

        model = build_graphsage_plus_model(X_fp.shape[1], X_onto.shape[1])
        counts = np.bincount(y_train.astype(int))
        class_weight = None
        if len(counts) == 2 and counts.min() > 0:
            total = counts.sum()
            class_weight = {0: float(total / (2 * counts[0])), 1: float(total / (2 * counts[1]))}
        model.fit({"fingerprint_input": Xfp_train, "atom_features_input": Xatoms_train, "adjacency_input": Xadj_train, "atom_mask_input": Xmasks_train, "ontology_input": Xonto_train}, y_train,
                  validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=18, mode="max", restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=7, mode="max", min_lr=1e-6)], verbose=1)

        probs = model.predict({"fingerprint_input": Xfp_test, "atom_features_input": Xatoms_test, "adjacency_input": Xadj_test, "atom_mask_input": Xmasks_test, "ontology_input": Xonto_test}, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
        metrics = {"test_accuracy": accuracy_score(y_test, preds), "test_f1": f1_score(y_test, preds, zero_division=0), "test_precision": precision_score(y_test, preds, zero_division=0), "test_recall": recall_score(y_test, preds, zero_division=0)}
        try:
            metrics["test_roc_auc"] = roc_auc_score(y_test, probs)
        except ValueError:
            metrics["test_roc_auc"] = 0.0
        mlflow.log_metrics(metrics)
        model.save(MODEL_PATH)

        meta_path = os.path.join(MODEL_DIR, "graphsage_plus_model_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": "OA-DeepFPLearn-GraphSAGE++", "metrics": metrics, "params": params}, f, indent=2)
        ontology_path = os.path.join(MODEL_DIR, "graphsage_plus_ontology_edges.json")
        with open(ontology_path, "w", encoding="utf-8") as f:
            json.dump(list(graph.edges()), f, indent=2)

        for artifact in [MODEL_PATH, autoencoder_path, encoder_path, meta_path, ontology_path]:
            mlflow.log_artifact(artifact)
        for artifact in save_visualizations(model, Xfp_test, Xatoms_test, Xadj_test, Xmasks_test, Xonto_test, df_test, y_test, probs):
            mlflow.log_artifact(artifact)
        print("\nFinal test metrics:")
        print(metrics)
        print(f"Saved GraphSAGE++ model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
