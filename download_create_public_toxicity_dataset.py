import os
import logging
import time
import requests
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")
logger = logging.getLogger("OA_DeepFPLearn")
INVALID_SMILES = []

OUTPUT_DIR = "data"
RAW_DIR = "data/raw"
OUTPUT_FILE = "data/toxicity_data_public.csv"

MOLECULENET_TOX21_URL = (
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz"
)

PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


PFAS_SEEDS = [
    ("PFOS", "1763-23-1", "PFAS_Sulfonate"),
    ("PFOA", "335-67-1", "PFAS_Carboxylate"),
    ("PFHxS", "355-46-4", "PFAS_Sulfonate"),
    ("PFBS", "375-73-5", "PFAS_Sulfonate"),
    ("PFHxA", "307-24-4", "PFAS_Carboxylate"),
    ("PFNA", "375-95-1", "PFAS_Carboxylate"),
    ("PFDA", "335-76-2", "PFAS_Carboxylate"),
    ("GenX", "13252-13-6", "PFAS_Ether"),
    ("ADONA", "958445-44-8", "PFAS_Ether"),
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
    name = str(name).upper()
    smiles = str(smiles)

    if "S(=O)(=O)" in smiles or name in ["PFOS", "PFHXS", "PFBS"]:
        return "PFAS_Sulfonate"
    if "C(=O)O" in smiles or name in ["PFOA", "PFHXA", "PFNA", "PFDA"]:
        return "PFAS_Carboxylate"
    if "OC" in smiles and smiles.count("F") >= 4:
        return "PFAS_Ether"
    if smiles.count("F") >= 4:
        return "PFAS"
    return "Non_PFAS"


def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return out_path

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def fetch_pubchem_smiles(identifier):
    url = (
        f"{PUBCHEM_PUG_REST}/compound/name/"
        f"{identifier}/property/CanonicalSMILES/JSON"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None

        data = r.json()
        props = data["PropertyTable"]["Properties"]
        return props[0].get("CanonicalSMILES")
    except Exception:
        return None


def create_pubchem_pfas_seed_dataset():
    rows = []

    for name, casrn, pfas_class in PFAS_SEEDS:
        smiles = fetch_pubchem_smiles(casrn)
        time.sleep(0.2)

        if smiles is None:
            smiles = fetch_pubchem_smiles(name)
            time.sleep(0.2)

        canonical = canonicalize_smiles(smiles) if smiles else None

        if canonical is None:
            continue

        rows.append({
            "dataset": "PubChem_PFAS_Seed",
            "compound_id": name,
            "smiles": canonical,
            "pfas_class": pfas_class,
            "toxicity": 1
        })

    return pd.DataFrame(rows)


def load_moleculenet_tox21():
    raw_path = os.path.join(RAW_DIR, "tox21.csv.gz")
    download_file(MOLECULENET_TOX21_URL, raw_path)

    df = pd.read_csv(raw_path)

    smiles_col = "smiles"
    id_col = "mol_id" if "mol_id" in df.columns else None

    task_cols = [c for c in df.columns if c not in ["smiles", "mol_id"]]

    rows = []

    for _, row in df.iterrows():
        smiles = canonicalize_smiles(row[smiles_col])
        if smiles is None:
            continue

        values = []
        for col in task_cols:
            if pd.notna(row[col]):
                values.append(float(row[col]))

        if not values:
            continue

        toxicity = int(max(values) > 0)
        compound_id = str(row[id_col]) if id_col else smiles
        pfas_class = infer_pfas_class(compound_id, smiles)

        rows.append({
            "dataset": "MoleculeNet_Tox21",
            "compound_id": compound_id,
            "smiles": smiles,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def load_local_epa_comptox_if_available():
    path = os.path.join(RAW_DIR, "epa_comptox.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    smiles_col = (
        lower.get("smiles")
        or lower.get("qsar_ready_smiles")
        or lower.get("canonical_smiles")
    )

    name_col = (
        lower.get("preferred_name")
        or lower.get("name")
        or lower.get("dtxsid")
        or lower.get("casrn")
    )

    class_col = (
        lower.get("pfas_class")
        or lower.get("pfas_category")
        or lower.get("chemical_class")
    )

    tox_col = (
        lower.get("toxicity")
        or lower.get("active")
        or lower.get("hitcall")
        or lower.get("hazard")
    )

    if smiles_col is None:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        smiles = canonicalize_smiles(row[smiles_col])
        if smiles is None:
            continue

        compound_id = str(row[name_col]) if name_col else smiles

        if class_col and pd.notna(row[class_col]):
            pfas_class = str(row[class_col]).replace(" ", "_")
        else:
            pfas_class = infer_pfas_class(compound_id, smiles)

        if tox_col and pd.notna(row[tox_col]):
            toxicity = int(float(row[tox_col]) > 0)
        else:
            toxicity = 1 if pfas_class != "Non_PFAS" else 0

        rows.append({
            "dataset": "EPA_CompTox",
            "compound_id": compound_id,
            "smiles": smiles,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def load_local_epa_toxcast_if_available():
    path = os.path.join(RAW_DIR, "epa_toxcast.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    smiles_col = lower.get("smiles") or lower.get("canonical_smiles")
    name_col = (
        lower.get("chemical_name")
        or lower.get("compound_id")
        or lower.get("casrn")
        or lower.get("name")
    )
    activity_col = (
        lower.get("hitcall")
        or lower.get("active")
        or lower.get("toxicity")
        or lower.get("response")
    )

    if smiles_col is None or activity_col is None:
        return pd.DataFrame()

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    grouped = df.groupby("canonical_smiles")[activity_col].max().reset_index()

    name_map = {}
    if name_col:
        name_map = df.groupby("canonical_smiles")[name_col].first().to_dict()

    rows = []

    for _, row in grouped.iterrows():
        smiles = row["canonical_smiles"]
        compound_id = str(name_map.get(smiles, smiles))
        toxicity = int(float(row[activity_col]) > 0)
        pfas_class = infer_pfas_class(compound_id, smiles)

        rows.append({
            "dataset": "EPA_ToxCast",
            "compound_id": compound_id,
            "smiles": smiles,
            "pfas_class": pfas_class,
            "toxicity": toxicity
        })

    return pd.DataFrame(rows)


def merge_clean_save(datasets):
    datasets = [d for d in datasets if d is not None and not d.empty]

    if not datasets:
        raise RuntimeError("No datasets were loaded.")

    df = pd.concat(datasets, ignore_index=True)

    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    df = df.dropna(subset=["smiles", "pfas_class", "toxicity"])

    df["dataset"] = df["dataset"].astype(str)
    df["compound_id"] = df["compound_id"].astype(str)
    df["pfas_class"] = df["pfas_class"].astype(str).str.replace(" ", "_")
    df["toxicity"] = df["toxicity"].astype(int)

    df = df.drop_duplicates(subset=["smiles", "pfas_class"], keep="first")
    df = df[["dataset", "compound_id", "smiles", "pfas_class", "toxicity"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    return df


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    datasets = []

    print("Downloading MoleculeNet Tox21...")
    datasets.append(load_moleculenet_tox21())

    print("Querying PubChem for PFAS seed compounds...")
    datasets.append(create_pubchem_pfas_seed_dataset())

    print("Checking local EPA CompTox file: data/raw/epa_comptox.csv")
    datasets.append(load_local_epa_comptox_if_available())

    print("Checking local EPA ToxCast file: data/raw/epa_toxcast.csv")
    datasets.append(load_local_epa_toxcast_if_available())

    final_df = merge_clean_save(datasets)
    invalid_path = export_invalid_smiles()

    print("\nCreated:", OUTPUT_FILE)
    print("Invalid SMILES log:", invalid_path)
    print("\nDataset counts:")
    print(final_df["dataset"].value_counts())

    print("\nPFAS class counts:")
    print(final_df["pfas_class"].value_counts())

    print("\nToxicity counts:")
    print(final_df["toxicity"].value_counts())


if __name__ == "__main__":
    main()