"""
Explainability analysis for OA-DeepFPLearn models.

Supported models:
- Encoder-DeepFPLearn: fingerprint + ontology inputs
- GNN-DeepFPLearn: fingerprint + atom graph + adjacency + ontology inputs
- GAT-DeepFPLearn: fingerprint + atom graph + adjacency + ontology inputs
- ChemBERTa-DeepFPLearn: fingerprint + ChemBERTa embeddings + ontology inputs

Outputs:
- Prediction explanation CSV
- Fusion attention weights CSV
- Ontology reasoning paths CSV
- SHAP summary CSV/PNG when SHAP is available
- Probability distribution plot

Example:
python explainability_analysis.py --model-type keras --model-path artifacts/oa_deepfplearn.keras --data-path data/toxicity_data.csv
python explainability_analysis.py --model-type gnn --model-path artifacts/oa_deepfplearn_gnn.keras --data-path data/toxicity_data.csv
python explainability_analysis.py --model-type gat --model-path artifacts/oa_deepfplearn_gat.keras --data-path data/toxicity_data.csv
python explainability_analysis.py --model-type chemberta --model-path artifacts/chemberta_oa_deepfplearn.keras --data-path data/toxicity_data.csv
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

from ontology_utils import (
    build_ontology_graph,
    train_ontology_embedding,
    get_ontology_vector,
    get_reasoning_path,
    infer_decision_actions,
)

FINGERPRINT_BITS = 2048
ONTOLOGY_DIM = 32
MAX_ATOMS = 80
ATOM_FEATURE_DIM = 10
MAX_SMILES_LENGTH = 128
CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
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


def smiles_to_morgan(smiles, n_bits=FINGERPRINT_BITS):
    mol = safe_mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    generator = MORGAN_GENERATOR if n_bits == FINGERPRINT_BITS else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_maccs(smiles):
    mol = safe_mol_from_smiles(smiles)
    arr = np.zeros((167,), dtype=np.float32)
    if mol is None:
        return arr
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprint(smiles, expected_dim=None):
    """Create fingerprint vector compatible with the loaded model.

    Supported dimensions:
    - 2048: Morgan only
    - 2215: Morgan + MACCS (2048 + 167), matching fingerprint_utils.py
    - any other value: Morgan with expected_dim bits as fallback
    """
    if expected_dim is None:
        expected_dim = FINGERPRINT_BITS
    expected_dim = int(expected_dim)
    if expected_dim == FINGERPRINT_BITS + 167:
        return np.concatenate([smiles_to_morgan(smiles, FINGERPRINT_BITS), smiles_to_maccs(smiles)]).astype(np.float32)
    if expected_dim == FINGERPRINT_BITS:
        return smiles_to_morgan(smiles, FINGERPRINT_BITS)
    return smiles_to_morgan(smiles, expected_dim)


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
        atom.GetMass() / 200.0,
    ], dtype=np.float32)


def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = safe_mol_from_smiles(smiles)
    X = np.zeros((max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    if mol is None:
        return X, A

    atoms = list(mol.GetAtoms())
    for i, atom in enumerate(atoms[:max_atoms]):
        X[i] = atom_features(atom)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            A[i, j] = 1.0
            A[j, i] = 1.0

    A += np.eye(max_atoms, dtype=np.float32)
    degree = np.sum(A, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return X, A_norm.astype(np.float32)


def load_chemberta_embeddings(smiles_list: List[str], batch_size: int = 16) -> np.ndarray:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Install ChemBERTa dependencies: pip install torch transformers")

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = [str(x) for x in smiles_list[start:start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SMILES_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            cls = output.last_hidden_state[:, 0, :]
            embeddings.append(cls.cpu().numpy())
    return np.vstack(embeddings).astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="OADeepFPLearn")
class GraphConv(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.w = self.add_weight(shape=(feature_dim, self.units), initializer="glorot_uniform", trainable=True, name="graph_conv_weight")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name="graph_conv_bias")
        super().build(input_shape)

    def call(self, inputs):
        X, A = inputs
        AX = tf.matmul(A, X)
        H = tf.matmul(AX, self.w) + self.b
        return self.activation(H) if self.activation is not None else H

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation_name})
        return config


@tf.keras.utils.register_keras_serializable(package="OADeepFPLearn")
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(feature_dim, self.units), initializer="glorot_uniform", trainable=True, name="gat_weight")
        self.a_src = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", trainable=True, name="gat_attention_source")
        self.a_dst = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", trainable=True, name="gat_attention_target")
        super().build(input_shape)

    def call(self, inputs, training=False):
        X, A = inputs
        H = tf.matmul(X, self.W)
        src_scores = tf.matmul(H, self.a_src)
        dst_scores = tf.matmul(H, self.a_dst)
        attention_logits = src_scores + tf.transpose(dst_scores, perm=[0, 2, 1])
        attention_logits = tf.nn.leaky_relu(attention_logits, alpha=0.2)
        mask = tf.where(A > 0, 0.0, -1e9)
        attention = tf.nn.softmax(attention_logits + mask, axis=-1)
        attention = self.dropout(attention, training=training)
        output = tf.matmul(attention, H)
        return self.activation(output) if self.activation is not None else output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate, "activation": self.activation_name})
        return config


def get_model_input_dim(model, input_name: str, default_dim: int) -> int:
    try:
        for inp in model.inputs:
            if inp.name.split(":")[0] == input_name:
                return int(inp.shape[-1])
    except Exception:
        pass
    return int(default_dim)


def prepare_inputs(data_path: str, model_type: str, sample_size: int = None, model=None):
    df = pd.read_csv(data_path).dropna(subset=["smiles", "pfas_class", "toxicity"]).reset_index(drop=True)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    graph = build_ontology_graph()
    ontology_embedding_model = train_ontology_embedding(graph, dimensions=ONTOLOGY_DIM)

    expected_fp_dim = get_model_input_dim(model, "fingerprint_input", FINGERPRINT_BITS) if model is not None else FINGERPRINT_BITS
    X_fp = np.array([smiles_to_fingerprint(s, expected_dim=expected_fp_dim) for s in df["smiles"]], dtype=np.float32)
    X_onto = np.array([get_ontology_vector(c, ontology_embedding_model, dimensions=ONTOLOGY_DIM) for c in df["pfas_class"]], dtype=np.float32)
    y = df["toxicity"].values.astype(np.float32)

    if model_type in {"gnn", "gat"}:
        graphs = [smiles_to_graph(s) for s in df["smiles"]]
        X_atoms = np.array([g[0] for g in graphs], dtype=np.float32)
        X_adj = np.array([g[1] for g in graphs], dtype=np.float32)
        inputs = {
            "fingerprint_input": X_fp,
            "atom_features_input": X_atoms,
            "adjacency_input": X_adj,
            "ontology_input": X_onto,
        }
    elif model_type == "chemberta":
        X_chem = load_chemberta_embeddings(df["smiles"].astype(str).tolist())
        inputs = {
            "fingerprint_input": X_fp,
            "chemberta_input": X_chem,
            "ontology_input": X_onto,
        }
    else:
        inputs = {
            "fingerprint_input": X_fp,
            "ontology_input": X_onto,
        }

    inputs = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in inputs.items()}
    return df, inputs, y, graph


def predict_and_save(model, inputs: Dict[str, np.ndarray], df: pd.DataFrame, out_dir: str) -> str:
    probs = model.predict(inputs, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    out = df.copy()
    out["predicted_probability"] = probs
    out["predicted_class"] = pred
    out["correct_prediction"] = (pred == out["toxicity"].astype(int)).astype(int)
    path = os.path.join(out_dir, "prediction_explanations.csv")
    out.to_csv(path, index=False)
    return path


def save_attention_weights(model, inputs: Dict[str, np.ndarray], out_dir: str) -> str:
    path = os.path.join(out_dir, "fusion_attention_weights.csv")
    try:
        attention_layer = model.get_layer("attention_weights")
        attention_model = tf.keras.Model(inputs=model.inputs, outputs=attention_layer.output)
        values = attention_model.predict(inputs, verbose=0)
        pd.DataFrame(values).to_csv(path, index=False)
    except Exception as exc:
        pd.DataFrame({"error": [str(exc)]}).to_csv(path, index=False)
    return path


def save_ontology_reasoning(df: pd.DataFrame, graph, out_dir: str) -> str:
    rows = []
    for _, row in df.iterrows():
        pfas_class = str(row["pfas_class"]).strip().replace(" ", "_")
        path_high = get_reasoning_path(graph, pfas_class, "High_Risk")
        path_medium = get_reasoning_path(graph, pfas_class, "Medium_Risk")
        actions = infer_decision_actions(graph, pfas_class)
        rows.append({
            "compound_id": row.get("compound_id", ""),
            "smiles": row["smiles"],
            "pfas_class": pfas_class,
            "path_to_high_risk": " -> ".join(path_high),
            "path_to_medium_risk": " -> ".join(path_medium),
            "recommended_actions": "; ".join(actions),
        })
    path = os.path.join(out_dir, "ontology_reasoning_paths.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_probability_plot(prediction_csv: str, out_dir: str) -> str:
    df = pd.read_csv(prediction_csv)
    plt.figure(figsize=(8, 5))
    plt.hist(df["predicted_probability"], bins=20)
    plt.xlabel("Predicted toxicity probability")
    plt.ylabel("Number of compounds")
    plt.title("Predicted Toxicity Probability Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, "predicted_probability_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def flatten_inputs(inputs: Dict[str, np.ndarray], max_graph_features: int = 512) -> Tuple[np.ndarray, List[str]]:
    arrays = []
    names = []
    for key, value in inputs.items():
        arr = value.reshape((value.shape[0], -1))
        if arr.shape[1] > max_graph_features and key in {"atom_features_input", "adjacency_input"}:
            arr = arr[:, :max_graph_features]
        arrays.append(arr)
        names.extend([f"{key}_{i}" for i in range(arr.shape[1])])
    return np.concatenate(arrays, axis=1), names


def save_shap_approximation(model, inputs: Dict[str, np.ndarray], out_dir: str, max_samples: int = 100) -> str:
    """Permutation SHAP on flattened features. Works for all model types but can be slow."""
    shap_csv = os.path.join(out_dir, "shap_feature_importance.csv")
    shap_png = os.path.join(out_dir, "shap_top_features.png")

    if not SHAP_AVAILABLE:
        pd.DataFrame({"message": ["SHAP not installed. Run: pip install shap"]}).to_csv(shap_csv, index=False)
        return shap_csv

    n = min(max_samples, next(iter(inputs.values())).shape[0])
    small_inputs = {k: v[:n] for k, v in inputs.items()}
    X_flat, feature_names = flatten_inputs(small_inputs)

    shapes = {k: v[:n].shape for k, v in small_inputs.items()}
    sizes = {k: int(np.prod(v.shape[1:])) for k, v in small_inputs.items()}
    keys = list(small_inputs.keys())

    def unflatten_predict(X):
        rebuilt = {}
        start = 0
        for key in keys:
            size = sizes[key]
            target_shape = shapes[key]
            block = X[:, start:start + size]
            # If graph features were truncated, pad back to original size.
            if block.shape[1] < size:
                pad = np.zeros((block.shape[0], size - block.shape[1]), dtype=np.float32)
                block = np.concatenate([block, pad], axis=1)
            rebuilt[key] = block.reshape((X.shape[0],) + target_shape[1:]).astype(np.float32)
            start += min(size, block.shape[1])
        return model.predict(rebuilt, verbose=0).ravel()

    try:
        background = shap.sample(X_flat, min(20, X_flat.shape[0]), random_state=42)
        explainer = shap.Explainer(unflatten_predict, background)
        values = explainer(X_flat[:min(50, X_flat.shape[0])])
        importance = np.abs(values.values).mean(axis=0)
        imp_df = pd.DataFrame({"feature": feature_names[:len(importance)], "mean_abs_shap": importance})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).head(50)
        imp_df.to_csv(shap_csv, index=False)

        plt.figure(figsize=(9, 7))
        plt.barh(imp_df["feature"].head(20)[::-1], imp_df["mean_abs_shap"].head(20)[::-1])
        plt.xlabel("Mean absolute SHAP value")
        plt.title("Top SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(shap_png, dpi=300)
        plt.close()
    except Exception as exc:
        pd.DataFrame({"error": [str(exc)]}).to_csv(shap_csv, index=False)

    return shap_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["keras", "gnn", "gat", "chemberta"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default="data/toxicity_data.csv")
    parser.add_argument("--output-dir", default="artifacts/explainability_full")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    custom_objects = {
        "GraphConv": GraphConv,
        "GraphAttentionLayer": GraphAttentionLayer,
    }
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)
    df, inputs, y, graph = prepare_inputs(args.data_path, args.model_type, args.sample_size, model=model)

    prediction_csv = predict_and_save(model, inputs, df, args.output_dir)
    attention_csv = save_attention_weights(model, inputs, args.output_dir)
    ontology_csv = save_ontology_reasoning(df, graph, args.output_dir)
    probability_png = save_probability_plot(prediction_csv, args.output_dir)

    shap_csv = None
    if not args.skip_shap:
        shap_csv = save_shap_approximation(model, inputs, args.output_dir)

    manifest = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "outputs": {
            "prediction_explanations": prediction_csv,
            "fusion_attention_weights": attention_csv,
            "ontology_reasoning_paths": ontology_csv,
            "probability_distribution": probability_png,
            "shap_feature_importance": shap_csv,
        },
    }
    manifest_path = os.path.join(args.output_dir, "explainability_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Explainability analysis completed.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
