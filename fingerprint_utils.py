import logging
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

logger = logging.getLogger("OA_DeepFPLearn")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS
)


def safe_mol_from_smiles(smiles):
    """Return a sanitized RDKit molecule or None for invalid/problematic SMILES."""
    if smiles is None:
        return None

    try:
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


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    arr = np.zeros((n_bits,), dtype=np.float32)
    mol = safe_mol_from_smiles(smiles)

    if mol is None:
        return arr

    if radius == MORGAN_RADIUS and n_bits == MORGAN_BITS:
        generator = MORGAN_GENERATOR
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits
        )

    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_maccs(smiles):
    arr = np.zeros((167,), dtype=np.float32)
    mol = safe_mol_from_smiles(smiles)

    if mol is None:
        return arr

    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprint(smiles):
    return np.concatenate([
        smiles_to_morgan(smiles),
        smiles_to_maccs(smiles)
    ]).astype(np.float32)
