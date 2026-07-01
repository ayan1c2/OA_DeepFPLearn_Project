import os
import logging
import re
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")
logger = logging.getLogger("OA_DeepFPLearn")
INVALID_SMILES = []

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "toxicity_data_public.csv")

PFAS_SEED_DATA = [
    ["PFAS_Demo", "PFOS", "C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F", "PFAS_Sulfonate", 1],
    ["PFAS_Demo", "PFOA", "C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O", "PFAS_Carboxylate", 1],
    ["PFAS_Demo", "GenX", "C(C(F)(F)OC(C(F)(F)F)(F)F)(C(=O)O)(F)F", "PFAS_Ether", 1],
    ["PFAS_Demo", "PFHxS", "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F", "PFAS_Sulfonate", 1],
    ["PFAS_Demo", "PFBS", "C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F", "PFAS_Sulfonate", 1],
    ["PFAS_Demo", "PFHxA", "C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)", "PFAS_Carboxylate", 1],
    ["PFAS_Demo", "PFNA", "C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(C(=O)O)", "PFAS_Carboxylate", 1],
    ["PFAS_Demo", "ADONA", "C(C(F)(F)OC(C(F)(F)F)(F)F)(C(=O)O)(F)F", "PFAS_Ether", 1],

    ["NonPFAS_Demo", "Ethanol", "CCO", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Acetone", "CC(=O)C", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Benzene", "c1ccccc1", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Toluene", "Cc1ccccc1", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Phenol", "c1ccc(cc1)O", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Water", "O", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Methane", "C", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Ethane", "CC", "Non_PFAS", 0],
    ["NonPFAS_Demo", "Chloroform", "ClC(Cl)Cl", "Non_PFAS", 1],
]


def canonicalize_smiles(smiles):
    try:
        if smiles is None or pd.isna(smiles):
            INVALID_SMILES.append(smiles)
            return None

        smiles = str(smiles).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            INVALID_SMILES.append(smiles)
            return None

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            INVALID_SMILES.append(smiles)
            return None

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception as exc:
        INVALID_SMILES.append(smiles)
        logger.warning("Invalid/problematic SMILES skipped: %s | %s", smiles, exc)
        return None


def export_invalid_smiles(output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "invalid_smiles.csv")
    pd.DataFrame({"invalid_smiles": sorted(set(map(str, INVALID_SMILES))) }).to_csv(path, index=False)
    return path


def infer_pfas_class(name, smiles):
    text = f"{name} {smiles}".upper()

    if "S(=O)(=O)O" in smiles or "SULFON" in text or name.upper() in ["PFOS", "PFHXS", "PFBS"]:
        return "PFAS_Sulfonate"

    if "C(=O)O" in smiles or "CARBOXY" in text or name.upper() in ["PFOA", "PFHXA", "PFNA", "PFDA"]:
        return "PFAS_Carboxylate"

    if "OC" in smiles and "F" in smiles:
        return "PFAS_Ether"

    if re.search(r"C\(F\)\(F\)", smiles) or smiles.count("F") >= 4:
        return "PFAS"

    return "Non_PFAS"


def load_seed_dataset():
    df = pd.DataFrame(
        PFAS_SEED_DATA,
        columns=["dataset", "compound_id", "smiles", "pfas_class", "toxicity"]
    )
    return df


def load_tox21_csv(path):
    """
    Expected input columns can be:
    compound_id/name, smiles, and one or more binary toxicity endpoints.

    Example compatible columns:
    compound_id, smiles, NR-AR, NR-ER, SR-p53, SR-MMP
    """

    df = pd.read_csv(path)

    smiles_col = find_column(df, ["smiles", "canonical_smiles", "SMILES"])
    name_col = find_column(df, ["compound_id", "name", "compound", "sample_id"])

    endpoint_cols = [
        col for col in df.columns
        if col not in [smiles_col, name_col]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows = []

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        canonical = canonicalize_smiles(smiles)

        if canonical is None:
            continue

        toxicity_values = [
            row[col] for col in endpoint_cols
            if pd.notna(row[col])
        ]

        if len(toxicity_values) == 0:
            continue

        toxicity = int(max(toxicity_values) > 0)

        compound_id = str(row[name_col]) if name_col else canonical
        pfas_class = infer_pfas_class(compound_id, canonical)

        rows.append({
            "dataset": "Tox21_Public",
            "compound_id": compound_id,
            "smiles": canonical,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def load_toxcast_csv(path):
    """
    Expected input columns can include:
    chemical_name, smiles, hitcall, modl_ga, endpoint, assay_component_endpoint_name.

    If several assays exist per chemical, this function aggregates them into one binary label.
    """

    df = pd.read_csv(path)

    smiles_col = find_column(df, ["smiles", "SMILES", "canonical_smiles"])
    name_col = find_column(df, ["chemical_name", "compound_id", "name", "chnm", "casrn"])
    activity_col = find_column(df, ["hitcall", "active", "toxicity", "response"])

    if smiles_col is None or activity_col is None:
        raise ValueError("ToxCast file must contain a SMILES column and an activity/hitcall column.")

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    grouped = (
        df.groupby("canonical_smiles")[activity_col]
        .max()
        .reset_index()
    )

    name_map = (
        df.groupby("canonical_smiles")[name_col]
        .first()
        .to_dict()
        if name_col else {}
    )

    rows = []

    for _, row in grouped.iterrows():
        smiles = row["canonical_smiles"]
        compound_id = str(name_map.get(smiles, smiles))
        toxicity = int(row[activity_col] > 0)
        pfas_class = infer_pfas_class(compound_id, smiles)

        rows.append({
            "dataset": "ToxCast_Public",
            "compound_id": compound_id,
            "smiles": smiles,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def load_comptox_csv(path):
    """
    Expected input columns can include:
    preferred_name, dtxsid, smiles, qsar_ready_smiles, pfas_category, toxicity.

    This function is flexible because exported CompTox files often vary by selected fields.
    """

    df = pd.read_csv(path)

    smiles_col = find_column(df, ["smiles", "qsar_ready_smiles", "canonical_smiles", "SMILES"])
    name_col = find_column(df, ["preferred_name", "compound_id", "name", "dtxsid", "casrn"])
    class_col = find_column(df, ["pfas_class", "pfas_category", "chemical_class"])
    tox_col = find_column(df, ["toxicity", "active", "hitcall", "hazard"])

    if smiles_col is None:
        raise ValueError("CompTox file must contain a SMILES column.")

    rows = []

    for _, row in df.iterrows():
        canonical = canonicalize_smiles(row[smiles_col])

        if canonical is None:
            continue

        compound_id = str(row[name_col]) if name_col else canonical

        if class_col and pd.notna(row[class_col]):
            pfas_class = str(row[class_col]).replace(" ", "_")
        else:
            pfas_class = infer_pfas_class(compound_id, canonical)

        if tox_col and pd.notna(row[tox_col]):
            toxicity = int(float(row[tox_col]) > 0)
        else:
            toxicity = 1 if pfas_class != "Non_PFAS" else 0

        rows.append({
            "dataset": "CompTox_Public",
            "compound_id": compound_id,
            "smiles": canonical,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def find_column(df, candidates):
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def clean_and_merge(datasets):
    df = pd.concat(datasets, ignore_index=True)

    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    df = df.dropna(subset=["smiles", "pfas_class", "toxicity"])

    df["compound_id"] = df["compound_id"].astype(str)
    df["pfas_class"] = df["pfas_class"].astype(str).str.strip().str.replace(" ", "_")
    df["toxicity"] = df["toxicity"].astype(int)

    df = (
        df.sort_values(["dataset", "compound_id"])
        .drop_duplicates(subset=["smiles", "pfas_class"], keep="first")
        .reset_index(drop=True)
    )

    return df[["dataset", "compound_id", "smiles", "pfas_class", "toxicity"]]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = [load_seed_dataset()]

    optional_files = {
        "data/raw_tox21.csv": load_tox21_csv,
        "data/raw_toxcast.csv": load_toxcast_csv,
        "data/raw_comptox.csv": load_comptox_csv
    }

    for path, loader in optional_files.items():
        if os.path.exists(path):
            print(f"Loading {path}")
            datasets.append(loader(path))
        else:
            print(f"Skipping {path}; file not found.")

    final_df = clean_and_merge(datasets)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCreated dataset: {OUTPUT_FILE}")
    print(final_df["dataset"].value_counts())
    print(final_df["pfas_class"].value_counts())
    print(final_df["toxicity"].value_counts())
    print(final_df.head())


if __name__ == "__main__":
    main()