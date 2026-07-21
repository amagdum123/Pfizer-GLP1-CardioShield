import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

from src.pipeline import explain_prediction, create_image


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="GLP1-CardioShield | hERG Assessment",
    page_icon="🧬",
    layout="wide"
)



# =====================================================
# Pfizer-Inspired Research Dashboard Styling
# =====================================================

# =====================================================
# Pfizer-Inspired Research Dashboard Styling
# =====================================================

st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

html,
body,
[class*="css"],
.stApp,
.stMarkdown,
.stText,
.stMetric,
.stButton button,
input,
textarea,
label {
    font-family: "Source Sans 3", Arial, sans-serif !important;
}


h1, h2, h3, h4 {
    font-family: "Source Sans 3", Arial, sans-serif !important;
}


/* Page background */
.stApp {
    background-color: #F7F9FC;
}


/* Main title */
h1 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 42px;
    font-weight: 700;
    color: #1C1C1C;
    letter-spacing: -1px;
}


/* Section headers */
h2 {
    font-size: 28px;
    font-weight: 600;
    color: #1C1C1C;
    font-family: 'Source Sans 3', sans-serif !important;
}

h3 {
    
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600;
    color: #1C1C1C;
}


/* Body text */
p {
    color: #4A4A4A;
    line-height: 1.6;
}


/* Pfizer-style information cards */
.metric-box {

    background: white;

    border: 1px solid #D9E2F2;

    border-radius: 4px;

    padding: 22px;

    min-height: 100px;

}


/* Research panels */
.research-panel {

    background: white;

    border-left: 4px solid #0093D0;

    padding: 18px;

    margin-bottom: 15px;

    border-radius: 3px;

}


/* Metadata text */
.metadata {

    color: #5F6B7A;

    font-size: 14px;

    letter-spacing: 0.3px;

}


/* Buttons */

.stButton button {

    background-color: #0093D0;

    color: white;

    border-radius: 3px;

    border: none;

    padding: 10px 28px;

    font-weight: 600;

}


.stButton button:hover {

    background-color: #006F9E;

}


/* Expander */

.streamlit-expanderHeader {

    font-weight: 600;

    color: #1C1C1C;

}


/* Data tables */

[data-testid="stDataFrame"] {

    border: 1px solid #D9E2F2;

}


/* Remove default Streamlit padding */

.block-container {

    padding-top: 3rem;

    padding-bottom: 3rem;

}
/* Force Streamlit font override */

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stMetric"],
[data-testid="stButton"],
[data-testid="stExpander"],
[data-testid="stDataFrame"],
.stMarkdown,
.stMarkdown p,
label,
input,
textarea,
button {
    font-family: "Source Sans 3", Arial, sans-serif !important;
}

</style>
""",
unsafe_allow_html=True
)



# =====================================================
# Research Header
# =====================================================


st.markdown(
"""
<div style="
border-bottom:1px solid #D9E2F2;
padding-bottom:20px;
margin-bottom:30px;
">

<h1 style="
margin-bottom:5px;
">
GLP1-CardioShield
</h1>


<p style="
font-size:20px;
color:#0B2E59;
margin-bottom:15px;
">
Machine Learning–Assisted Cardiotoxicity Screening Platform
</p>


<p class="metadata">

Computational Chemistry | Machine Learning Screening | Explainable AI

</p>


</div>

""",
unsafe_allow_html=True
)



# =====================================================
# Model Information Banner
# =====================================================


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
    """
    <div class="metric-box">

    <b>Prediction Endpoint</b>

    <br>

    hERG Ion Channel Liability

    </div>
    """,
    unsafe_allow_html=True
    )



with col2:

    st.markdown(
    """
    <div class="metric-box">

    <b>Machine Learning Model</b>

    <br>

    Combined Random Forest

    </div>
    """,
    unsafe_allow_html=True
    )



with col3:

    st.markdown(
    """
    <div class="metric-box">

    <b>Validation Performance</b>

    <br>

    ROC-AUC: 0.931

    </div>
    """,
    unsafe_allow_html=True
    )



st.divider()



# =====================================================
# Molecule Input
# =====================================================


st.subheader(
    "Compound Input"
)


st.markdown(
"""
Enter a molecular SMILES representation to evaluate
predicted hERG cardiotoxicity risk.
"""
)


smiles = st.text_input(
    "SMILES Structure",
    value="CCOO",
    placeholder="Enter molecular SMILES..."
)



with st.expander(
    "Example compound"
):

    st.code(
        "CCN(CC)CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    )

    # =====================================================
# Risk Assessment Execution
# =====================================================


if st.button(
    "Evaluate Compound",
    type="primary"
):

    try:

        probability, top_features, bit_info, mol = explain_prediction(smiles)


        risk_percent = probability * 100



        # ---------------------------------------------
        # Risk Classification
        # ---------------------------------------------


        if risk_percent < 20:

            risk_label = "Low Predicted Liability"
            risk_color = "#2E7D32"


        elif risk_percent < 50:

            risk_label = "Moderate Predicted Liability"
            risk_color = "#B26A00"


        else:

            risk_label = "High Predicted Liability"
            risk_color = "#B3261E"



        # ---------------------------------------------
        # Assessment Summary
        # ---------------------------------------------


        st.subheader(
            "Compound Assessment"
        )


        col1, col2, col3 = st.columns(3)



        with col1:

            st.markdown(
            f"""
            <div class="metric-box">

            <b>Predicted hERG Probability</b>

            <h2>
            {risk_percent:.1f}%
            </h2>

            </div>
            """,
            unsafe_allow_html=True
            )



        with col2:

            st.markdown(
            f"""
            <div class="metric-box">

            <b>Classification</b>

            <h3 style="
            color:{risk_color};
            ">
            {risk_label}
            </h3>

            </div>
            """,
            unsafe_allow_html=True
            )



        with col3:

            st.markdown(
            """
            <div class="metric-box">

            <b>Prediction Model</b>

            <h3>
            Combined RF
            </h3>

            </div>
            """,
            unsafe_allow_html=True
            )



        st.divider()



        # ---------------------------------------------
        # Molecular Analysis
        # ---------------------------------------------


        st.subheader(
            "Structural Analysis"
        )


        left, right = st.columns(
            [1,1]
        )



        with left:


            st.markdown(
            """
            <div class="research-panel">

            <b>Molecular Representation</b>

            <br><br>

            Highlighted regions indicate molecular fragments
            contributing strongly to the prediction.

            </div>
            """,
            unsafe_allow_html=True
            )


            img_bytes = create_image(
                mol,
                bit_info,
                top_features
            )


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            ) as tmp:

                tmp.write(img_bytes)

                tmp_path = tmp.name



            image = Image.open(
                tmp_path
            )


            st.image(
                image,
                use_container_width=True
            )


            os.unlink(
                tmp_path
            )



        with right:


            st.markdown(
            """
            <div class="research-panel">

            <b>Prediction Drivers</b>

            <br><br>

            Features identified by SHAP explainability analysis.

            </div>
            """,
            unsafe_allow_html=True
            )



            readable = top_features.copy()



            name_map = {

                "logP":
                    "Lipophilicity (logP)",

                "molecular_weight":
                    "Molecular Weight",

                "tpsa":
                    "Polar Surface Area",

                "hbd":
                    "Hydrogen Bond Donors",

                "hba":
                    "Hydrogen Bond Acceptors",

                "rotatable_bonds":
                    "Molecular Flexibility",

                "num_rings":
                    "Ring Count",

                "aromatic_rings":
                    "Aromatic Rings"

            }



            readable["feature"] = readable[
                "feature"
            ].astype(str).map(
                lambda x:
                name_map.get(
                    x,
                    f"Morgan Fragment {x}"
                )
            )



            readable["Effect"] = readable[
                "impact"
            ].apply(
                lambda x:
                "Increases liability"
                if x > 0
                else
                "Decreases liability"
            )



            table = readable[
                [
                    "feature",
                    "impact",
                    "Effect"
                ]
            ].rename(
                columns={

                    "feature":
                    "Feature",

                    "impact":
                    "SHAP Impact"

                }
            )


            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )



        st.divider()



        # ---------------------------------------------
        # Explainability
        # ---------------------------------------------


        st.subheader(
            "Model Explainability"
        )


        st.markdown(
        """
        SHAP values quantify how individual molecular
        features influenced the final model prediction.
        """
        )



        chart_df = top_features.sort_values(
            "impact"
        )



        labels = []


        for f in chart_df["feature"].astype(str):

            labels.append(
                name_map.get(
                    f,
                    f"Morgan Fragment {f}"
                )
            )



        fig, ax = plt.subplots(
            figsize=(8,5)
        )


        ax.barh(
            labels,
            chart_df["impact"],
            color="#0093D0"

        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        ax.grid(
            axis="x",
            alpha=0.2
        )


        ax.set_xlabel(
            "SHAP contribution"
        )


        ax.set_title(
            "Top Molecular Features"
        )


        st.pyplot(
            fig,
            use_container_width=True
        )



        st.divider()



        # ---------------------------------------------
        # Interpretation
        # ---------------------------------------------


        st.subheader(
            "Scientific Interpretation"
        )



        positive = readable[
            readable["impact"] > 0
        ].head(3)



        negative = readable[
            readable["impact"] < 0
        ].head(3)



        if len(positive) > 0:

            st.markdown(
            """
            **Features associated with increased predicted liability**
            """
            )


            for feature in positive["feature"]:

                st.write(
                    "• " + feature
                )



        if len(negative) > 0:

            st.markdown(
            """
            **Features associated with decreased predicted liability**
            """
            )


            for feature in negative["feature"]:

                st.write(
                    "• " + feature
                )



        st.info(
        """
        GLP1-CardioShield provides machine learning-based estimates
        of potential hERG liability for research purposes.
        Predictions should not replace experimental validation.
        """
        )



    except Exception as e:


        st.error(
            f"Unable to analyze compound: {e}"
        )



# =====================================================
# Footer
# =====================================================


st.divider()


st.markdown(
"""
<p class="metadata">

GLP1-CardioShield Research Prototype

<br>

Built using RDKit, Scikit-learn, SHAP, Streamlit

</p>
""",
unsafe_allow_html=True
)