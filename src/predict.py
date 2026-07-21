import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import AllChem

MODEL_PATH = "models/random_forest_combined_herg.pkl"
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    features = {}
    features["molecular_weight"] = Descriptors.MolWt(mol)
    features["logP"] = Crippen.MolLogP(mol)
    features["hbd"] = Lipinski.NumHDonors(mol)
    features["hba"] = Lipinski.NumHAcceptors(mol)
    features["rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
    features["tpsa"] = rdMolDescriptors.CalcTPSA(mol)
    features["num_rings"] = Lipinski.RingCount(mol)
    features["aromatic_rings"] = Lipinski.NumAromaticRings(mol)
    return features


def calculate_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )
    return list(fp)


def predict(smiles):
    model = joblib.load(MODEL_PATH)
    # descriptor features
    descriptors = calculate_descriptors(smiles)
    desc_df = pd.DataFrame([descriptors])
    # Morgan fingerprints
    fp = calculate_morgan(smiles)
    fp_df = pd.DataFrame(
        [fp],
        columns=[str(i) for i in range(2048)]
    )
    # combine exactly like training
    combined = pd.concat(
        [desc_df, fp_df],
        axis=1
    )
    probability = model.predict_proba(combined)[0][1]
    return probability
if __name__ == "__main__":
    smiles = input("Enter SMILES: ")
    risk = predict(smiles)
    print("\nhERG toxicity probability:")
    print(f"{risk:.2%}")
    if risk > 0.5:
        print("Prediction: Potential hERG blocker")
    else:
        print("Prediction: Not a hERG blocker")