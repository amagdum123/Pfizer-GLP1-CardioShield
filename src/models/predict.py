import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


# Load model
model = joblib.load("models/random_forest_herg.pkl")

print("Model loaded!")


def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    features = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "num_rings": Lipinski.RingCount(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
    }

    return pd.DataFrame([features])


# Example molecule
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin

X = featurize_smiles(smiles)

prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]


print("\nPrediction:")
print("hERG blocker" if prediction == 1 else "Not a hERG blocker")
print(f"Risk probability: {probability:.2%}")
