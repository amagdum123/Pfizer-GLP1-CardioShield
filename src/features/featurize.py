import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors

#load HERG dataset
input_path = "data/herg.csv"
output_path = "data/herg_features.csv"
df = pd.read_csv(input_path)
print("Loaded dataset:")
print(df.head())
print("Shape:", df.shape)
def calculate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    features = {}

    #chemical molecular descriptors of toxicity 
    features["molecular_weight"] = Descriptors.MolWt(mol)
    features["logP"] = Crippen.MolLogP(mol)
    features["hbd"] = Lipinski.NumHDonors(mol)
    features["hba"] = Lipinski.NumHAcceptors(mol)
    features["rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
    features["tpsa"] = rdMolDescriptors.CalcTPSA(mol)
    features["num_rings"] = Lipinski.RingCount(mol)
    features["aromatic_rings"] = Lipinski.NumAromaticRings(mol)

    return features


#feature list
feature_list = []
for smiles in df["SMILES"]:
    feature_list.append(calculate_features(smiles))
features_df = pd.DataFrame(feature_list)

#concatenate with labels
result = pd.concat(
    [features_df, df["Y"]],
    axis=1
)

#remove failed molecules
result = result.dropna()
print("\nFeature dataset:")
print(result.head())
print("Shape:", result.shape)


#save
result.to_csv(output_path, index=False)
print(f"\nSaved features to {output_path}")
