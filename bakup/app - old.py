import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from fingerprint_utils import smiles_to_fingerprint
from ontology_utils import (
    build_ontology_graph,
    train_ontology_embedding,
    get_ontology_vector,
    get_reasoning_path,
    infer_decision_actions,
)

MODEL_PATH = "artifacts/oa_deepfplearn.keras"
INDEX_HTML_PATH = "index.html"
ONTOLOGY_DIM = 32

KNOWN_PFAS = {
    "PFOS-like": {"class": "PFAS_Sulfonate", "status": "Regulated"},
    "PFOA-like": {"class": "PFAS_Carboxylate", "status": "Regulated"},
    "GenX-like": {"class": "PFAS_Ether", "status": "Emerging Concern"},
}

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    graph = build_ontology_graph()
    emb = train_ontology_embedding(graph, dimensions=ONTOLOGY_DIM)
    return model, graph, emb

def identify_compound(smiles):
    s = str(smiles)
    if "S(=O)(=O)" in s:
        return "PFOS-like"
    if "C(=O)O" in s:
        return "PFOA-like"
    if "OC" in s and s.count("F") >= 4:
        return "GenX-like"
    return "Unknown Compound"

def classify_pfas(smiles):
    name = identify_compound(smiles)
    return KNOWN_PFAS.get(name, {}).get("class", "Non_PFAS")

def molecular_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2)
    }

def render_home():
    p = Path(INDEX_HTML_PATH)
    if p.exists():
        components.html(p.read_text(encoding="utf-8"), height=1200, scrolling=True)
    else:
        st.info("index.html not found")

st.set_page_config(page_title="OA-DeepFPLearn Platform", page_icon="🧪", layout="wide")

page = st.sidebar.radio(
    "Navigation",
    ["Platform Overview", "Prediction Tool", "How It Works"]
)

if page == "Platform Overview":
    render_home()

elif page == "Prediction Tool":
    st.title("Ontology-Aware DeepFPLearn")

    smiles = st.text_area(
        "SMILES",
        "C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F"
    )

    if st.button("Analyze"):
        model, graph, ontology_embedding = load_resources()

        compound = identify_compound(smiles)
        chem_class = classify_pfas(smiles)
        info = molecular_info(smiles)

        fp = smiles_to_fingerprint(smiles).reshape(1, -1)
        onto = get_ontology_vector(
            chem_class,
            ontology_embedding,
            dimensions=ONTOLOGY_DIM
        ).reshape(1, -1)

        probability = float(
            model.predict(
                {
                    "fingerprint_input": fp,
                    "ontology_input": onto
                },
                verbose=0
            )[0][0]
        )

        st.subheader("Compound Information")
        st.write(f"**Identified Compound:** {compound}")
        st.write(f"**Chemical Class:** {chem_class}")

        if info:
            st.write(f"**Formula:** {info['formula']}")
            st.write(f"**Molecular Weight:** {info['molecular_weight']} g/mol")

        status = KNOWN_PFAS.get(compound, {}).get("status", "Unknown")
        st.write(f"**Regulatory Status:** {status}")

        st.subheader("Prediction")
        st.metric("Toxicity Probability", f"{probability:.3f}")

        if probability >= 0.5:
            st.error("Predicted Toxic / Active")
        else:
            st.success("Predicted Non‑Toxic / Inactive")

        risk = "High Risk" if probability >= 0.7 else "Moderate Risk" if probability >= 0.4 else "Low Risk"
        st.write(f"**Risk Category:** {risk}")

        path = get_reasoning_path(graph, chem_class, "High_Risk")
        st.subheader("Ontology Explainability")
        st.code(" -> ".join(path) if path else chem_class)

        actions = infer_decision_actions(graph, chem_class)
        st.subheader("Recommended Actions")
        for a in actions:
            st.write("•", a)

else:
    st.title("How It Works")
    st.markdown("""
    SMILES → RDKit → Compound Identification → PFAS Classification → Ontology Graph Lookup
    → Node2Vec Ontology Embedding → OA‑DeepFPLearn → Toxicity Prediction
    → Risk Category → Regulatory Status → Recommended Actions
    """)
