import pandas as pd
import joblib
import shap
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    AllChem,
    Draw
)


MODEL_PATH = "models/random_forest_combined_herg.pkl"

def calculate_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")


    descriptors = {

        "molecular_weight":
            Descriptors.MolWt(mol),

        "logP":
            Crippen.MolLogP(mol),

        "hbd":
            Lipinski.NumHDonors(mol),

        "hba":
            Lipinski.NumHAcceptors(mol),

        "rotatable_bonds":
            Lipinski.NumRotatableBonds(mol),

        "tpsa":
            rdMolDescriptors.CalcTPSA(mol),

        "num_rings":
            Lipinski.RingCount(mol),

        "aromatic_rings":
            Lipinski.NumAromaticRings(mol)

    }


    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )


    fingerprint = list(fp)


    fp_df = pd.DataFrame(
        [fingerprint],
        columns=[
            str(i)
            for i in range(2048)
        ]
    )


    desc_df = pd.DataFrame(
        [descriptors]
    )


    features = pd.concat(
        [
            desc_df,
            fp_df
        ],
        axis=1
    )


    return features



def draw_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")


    return Draw.MolToImage(
        mol,
        size=(500,500)
    )



def explain(smiles):

    model = joblib.load(
        MODEL_PATH
    )


    features = calculate_features(
        smiles
    )


    prediction = model.predict_proba(
        features
    )[0][1]



    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        features,
        check_additivity=False
    )


    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif len(shap_values.shape) == 3:

        values = shap_values[0][:,1]

    else:

        values = shap_values[0]



    values = np.array(
        values
    ).flatten()



    importance = pd.DataFrame({

        "feature":
            features.columns,

        "impact":
            values

    })


    importance["abs"] = abs(
        importance["impact"]
    )


    # only show active features
    importance = importance[
        features.iloc[0].values != 0
    ]


    top = (
        importance
        .sort_values(
            "abs",
            ascending=False
        )
        .head(10)
    )



    readable = []


    names = {

        "logP":
            "Lipophilicity (logP)",

        "molecular_weight":
            "Molecular weight",

        "tpsa":
            "Polar surface area",

        "hbd":
            "Hydrogen bond donors",

        "hba":
            "Hydrogen bond acceptors",

        "rotatable_bonds":
            "Molecular flexibility",

        "num_rings":
            "Ring structures",

        "aromatic_rings":
            "Aromatic rings"

    }



    for _, row in top.iterrows():

        feature = row["feature"]

        impact = row["impact"]


        if str(feature).isdigit():

            feature_name = (
                f"Morgan fingerprint fragment {feature}"
            )

        else:

            feature_name = names.get(
                feature,
                feature
            )



        readable.append({

            "Feature":
                feature_name,

            "Impact":
                round(
                    impact,
                    3
                ),

            "Effect":
                (
                    "⬆️ Increases risk"
                    if impact > 0
                    else
                    "⬇️ Decreases risk"
                )

        })


    explanation = pd.DataFrame(
        readable
    )
    return prediction, explanation