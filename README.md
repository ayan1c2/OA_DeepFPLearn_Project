# Ontology-Aware DeepFPLearn (OA-DeepFPLearn) for Explainable PFAS Toxicity Prediction

This repository contains the complete implementation of the **Ontology-Aware DeepFPLearn (OA-DeepFPLearn)** framework for explainable **Per- and Polyfluoroalkyl Substances (PFAS)** toxicity prediction.

The framework integrates **molecular representation learning**, **knowledge graph engineering**, **ontology embeddings**, **semantic reasoning**, and **explainable artificial intelligence (XAI)** to provide accurate, interpretable, and scalable toxicity prediction for environmental risk assessment.

The implementation accompanies the research article:

> **Ontology-Aware DeepFPLearn Architectures for Explainable PFAS Toxicity Prediction**

---

# Framework Overview

The proposed framework consists of four major components:

1. **Data Preparation**
   - MoleculeNet Tox21
   - PubChem
   - EPA CompTox
   - EPA ToxCast
   - RDKit molecular preprocessing
   - Canonical SMILES generation
   - PFAS classification

2. **Knowledge Engineering**
   - PFAS ontology construction
   - RDF/OWL knowledge graph
   - Turtle (TTL) serialization
   - Node2Vec ontology embeddings
   - SPARQL querying
   - HermiT ontology reasoning

3. **Ontology-Aware Deep Learning**
   - AutoEncoder-DeepFPLearn
   - GNN-DeepFPLearn
   - GAT-DeepFPLearn
   - GraphSAGE-DeepFPLearn
   - GraphSAGE++-DeepFPLearn
   - ChemBERTa-DeepFPLearn

4. **Explainability and Decision Support**
   - SHAP feature attribution
   - Attention-weight visualization
   - Ontology reasoning paths
   - Interactive Streamlit interface

---

# Project Structure

```text
OA_DeepFPLearn/
│
├── app.py
├── train_autoencoder.py
├── train_gnn.py
├── train_gat.py
├── train_graphsage.py
├── train_graphsageplusplus.py
├── train_chemberta.py
├── compare_models.py
├── explainability_analysis.py
├── ontology_utils.py
├── fingerprint_utils.py
├── logger_utils.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── toxicity_data.csv
│
├── ontology/
│   ├── pfas_ontology.ttl
│   ├── pfas_instances.ttl
│   ├── pfas_ontology.owl
│   └── queries/
│
├── artifacts/
│
├── explainability/
│
├── results/
│
└── models/
```

---

# Software Requirements

- Python 3.10+
- TensorFlow / Keras
- PyTorch
- PyTorch Geometric
- RDKit
- NetworkX
- Node2Vec
- RDFLib
- Protégé
- MLflow
- Streamlit
- SHAP
- HuggingFace Transformers
- Scikit-learn

---

# Installation

```bash
git clone https://github.com/<username>/OA-DeepFPLearn.git

cd OA-DeepFPLearn

python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

For RDKit (recommended)

```bash
conda install -c conda-forge rdkit
```

---

# Dataset Preparation

Generate the unified PFAS toxicity dataset

```bash
python prepare_dataset.py
```

Output

```
data/toxicity_data.csv
```

---

# Ontology Construction

Generate the PFAS ontology

```bash
python ontology_utils.py
```

Generated files include

```
pfas_ontology.ttl
pfas_ontology.owl
pfas_ontology.graphml
pfas_ontology.json
pfas_ontology_visualization.png
```

---

# Model Training

Example

### AutoEncoder-DeepFPLearn

```bash
python train_autoencoder.py
```

### GNN-DeepFPLearn

```bash
python train_gnn.py
```

### GAT-DeepFPLearn

```bash
python train_gat.py
```

### GraphSAGE

```bash
python train_graphsage.py
```

### GraphSAGE++

```bash
python train_graphsageplusplus.py
```

### ChemBERTa

```bash
python train_chemberta.py
```

---

# MLflow Experiment Tracking

Start MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open

```
http://127.0.0.1:5000
```

MLflow records

- Hyperparameters
- Training metrics
- Validation metrics
- Models
- Explainability artifacts
- Dataset statistics
- Ontology outputs

---

# Explainability Analysis

Generate explanations

```bash
python explainability_analysis.py \
--model-type keras \
--model-path artifacts/oa_deepfplearn.keras
```

Outputs include

- Prediction explanations
- SHAP feature importance
- Attention weights
- Ontology reasoning paths
- Toxicity probability distributions

---

# Streamlit Application

Launch

```bash
streamlit run app.py
```

The application supports

- Molecular SMILES input
- Automatic PFAS identification
- Toxicity prediction
- Molecular descriptors
- Ontology visualization
- Semantic reasoning
- Confidence estimation
- Regulatory interpretation
- Decision-support recommendations

---

# Technologies

- TensorFlow / Keras
- PyTorch
- PyTorch Geometric
- RDKit
- HuggingFace Transformers
- NetworkX
- Node2Vec
- RDFLib
- Protégé
- HermiT
- SPARQL
- MLflow
- Streamlit

---

# Research Contributions

The repository implements

- Ontology-aware DeepFPLearn for PFAS toxicity prediction
- Six ontology-aware deep learning architectures
- PFAS ontology engineering using RDF/OWL
- Ontology embedding through Node2Vec
- Explainable AI using SHAP and attention mechanisms
- Semantic reasoning and SPARQL querying
- Interactive environmental decision-support platform
- Fully reproducible experimental pipeline

---

# Citation

If you use this repository, please cite:

```
Ayan Chatterjee,
Ontology-Aware DeepFPLearn Architectures for Explainable PFAS Toxicity Prediction,
2026.
```

---

# License

MIT License

---

# Contact

**Ayan Chatterjee**

Department of Digital Technology

NILU – Norwegian Institute for Air Research

Email: ayan@nilu.no