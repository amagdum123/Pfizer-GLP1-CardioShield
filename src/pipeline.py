import os

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


# =====================================================
# Model Path
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "random_forest_combined_herg.pkl"
)


# =====================================================
# Feature Calculation
# =====================================================

def calculate_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    descriptors = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "num_rings": Lipinski.RingCount(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol)
    }

    bit_info = {}

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
        bitInfo=bit_info
    )

    fingerprint = list(fp)

    fp_df = pd.DataFrame(
        [fingerprint],
        columns=[str(i) for i in range(2048)]
    )

    desc_df = pd.DataFrame(
        [descriptors]
    )

    features = pd.concat(
        [desc_df, fp_df],
        axis=1
    )

    return features, bit_info, mol


# =====================================================
# Model Prediction + SHAP Explainability
# =====================================================

def explain_prediction(smiles):

    model = joblib.load(MODEL_PATH)

    features, bit_info, mol = calculate_features(smiles)

    probability = model.predict_proba(
        features
    )[0][1]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        features,
        check_additivity=False
    )

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif len(shap_values.shape) == 3:

        values = shap_values[0][:, 1]

    else:

        values = shap_values[0]

    values = np.array(
        values
    ).flatten()

    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "impact": values,
            "abs": np.abs(values)
        }
    )

    # Only keep features that are active
    active = features.columns[
        features.iloc[0] != 0
    ]

    importance = importance[
        importance["feature"].isin(active)
    ]

    top = (
        importance
        .sort_values(
            "abs",
            ascending=False
        )
        .head(10)
    )

    return (
        probability,
        top,
        bit_info,
        mol
    )


# =====================================================
# Molecule Visualization
# =====================================================

def create_image(
    mol,
    bit_info,
    top_features
):

    highlight_atoms = []

    for feature in top_features["feature"]:

        if str(feature).isdigit():

            bit = int(feature)

            if bit in bit_info:

                for atom_radius in bit_info[bit]:

                    highlight_atoms.append(
                        atom_radius[0]
                    )

    highlight_atoms = list(
        set(highlight_atoms)
    )

    # RDKit's standard image renderer.
    # This avoids rdMolDraw2D and its
    # libXrender dependency on Streamlit Cloud.

    image = Draw.MolToImage(
        mol,
        size=(500, 500),
        highlightAtoms=highlight_atoms
    )

    return image