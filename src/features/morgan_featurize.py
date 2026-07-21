import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

df = pd.read_csv("data/herg.csv")
print("Loaded dataset:")
print(df.head())


#fingerprint generator
generator = GetMorganGenerator(
    radius=2,
    fpSize=2048
)
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    return list(fp)

#generate fingerprints
fingerprints = []
for smiles in df["SMILES"]:
    fingerprints.append(smiles_to_fp(smiles))
fp_df = pd.DataFrame(fingerprints)


fp_df["Y"] = df["Y"]
print("\nFingerprint dataset:")
print(fp_df.head())
print("\nShape:")
print(fp_df.shape)
fp_df.to_csv(
    "data/herg_morgan_features.csv",
    index=False
)
print("\nSaved Morgan fingerprints!")