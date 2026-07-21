import streamlit as st
import sys
import matplotlib.pyplot as plt

sys.path.append("src")
from explain import explain, draw_molecule

st.title(
    "🧬 CardioShield: Explainable hERG Toxicity Predictor"
)

st.write(
    """
    Predict cardiac hERG toxicity risk from a SMILES string
    using a tuned Random Forest model with molecular descriptors
    and Morgan fingerprints.
    """
)

smiles = st.text_input(
    "Enter SMILES:",
    "CCOO"
)

if st.button("Predict"):

    probability, explanation = explain(
        smiles
    )

    st.subheader(
        "Prediction"
    )

    st.metric(
        "hERG Risk Probability",
        f"{probability:.2%}"
    )

    if probability < 0.5:

        st.success(
            "🟢 Low predicted hERG inhibition risk"
        )
    else:

        st.error(
            "🔴 High predicted hERG inhibition risk"
        )

    st.subheader(
        "Molecule Structure"
    )

    st.image(
        draw_molecule(smiles),
        caption="Input molecule"
    )


    st.subheader(
        "Explainability"
    )

    st.dataframe(
        explanation,
        hide_index=True
    )


    st.subheader(
        "Risk Factors"
    )

    chart = explanation.sort_values(
        "Impact"
    )

    fig, ax = plt.subplots()

    ax.barh(
        chart["Feature"],
        chart["Impact"]
    )
    ax.set_xlabel(
        "SHAP contribution"
    )

    st.pyplot(
        fig
    )
    st.divider()
    st.caption(
        """
        Model: Tuned Random Forest  
        Features: Morgan fingerprints + molecular descriptors  
        Target: hERG cardiac toxicity prediction
        """
    )