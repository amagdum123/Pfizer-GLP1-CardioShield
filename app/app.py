import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt

from src.pipeline import explain_prediction, create_image


# =====================================================
# Logo
# =====================================================

# CardioShield.png sits one level up from this file (app/app.py -> ../CardioShield.png)
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "CardioShield.png",
)


def get_logo_base64(path: str) -> str | None:
    """Read the logo file and return a base64 string, or None if it can't be found."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


logo_b64 = get_logo_base64(LOGO_PATH)


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="GLP1-CardioShield | hERG Assessment",
    page_icon=Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🧬",
    layout="wide",
)


# =====================================================
# Pfizer-Inspired Research Dashboard Styling
# =====================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --accent: #0093D0;
        --accent-dark: #006F9E;
        --ink: #1C1C1C;
        --ink-soft: #4A4A4A;
        --line: #D9E2F2;
        --surface: #FFFFFF;
        --bg: #F7F9FC;
    }

    html, body, [class*="css"],
    .stApp, .stMarkdown, .stText, .stMetric,
    .stButton button, input, textarea, label,
    h1, h2, h3, h4 {
        font-family: "Source Sans 3", Arial, sans-serif !important;
    }

    /* Page background */
    .stApp {
        background-color: var(--bg);
    }

    /* Main title */
    h1 {
        font-size: 42px;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -1px;
    }

    /* Section headers */
    h2 {
        font-size: 28px;
        font-weight: 600;
        color: var(--ink);
    }

    h3 {
        font-weight: 600;
        color: var(--ink);
    }

    /* Body text */
    p {
        color: var(--ink-soft);
        line-height: 1.6;
    }

    /* Pfizer-style information cards */
    .metric-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 22px;
        min-height: 100px;
    }

    /* Research panels */
    .research-panel {
        background: var(--surface);
        border-left: 4px solid var(--accent);
        padding: 18px;
        margin-bottom: 15px;
        border-radius: 3px;
    }

    /* Metadata text */
    .metadata {
        color: var(--ink-soft);
        font-size: 14px;
        letter-spacing: 0.3px;
    }

    /* Buttons */
    .stButton button {
        background-color: var(--accent);
        color: white;
        border-radius: 3px;
        border: none;
        padding: 10px 28px;
        font-weight: 600;
    }

    .stButton button:hover {
        background-color: var(--accent-dark);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--ink);
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
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
    .stMarkdown, .stMarkdown p,
    label, input, textarea, button {
        font-family: "Source Sans 3", Arial, sans-serif !important;
    }

    /* Text input: swap Streamlit's default red focus ring for the accent color */
    .stTextInput div[data-baseweb="base-input"],
    .stTextInput div[data-baseweb="input"],
    .stTextInput div[data-baseweb="base-input"] > div,
    .stTextInput input {
        border-color: var(--line) !important;
        box-shadow: none !important;
        outline: none !important;
        caret-color: var(--accent) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .stTextInput div[data-baseweb="base-input"]:focus-within,
    .stTextInput div[data-baseweb="input"]:focus-within,
    .stTextInput div[data-baseweb="base-input"]:has(input:focus) {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .stTextInput input:focus,
    .stTextInput input:focus-visible {
        box-shadow: none !important;
        outline: none !important;
        border-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Research Header
# =====================================================

logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height:56px; margin-right:18px;">'
    if logo_b64
    else ""
)

st.markdown(
    f"""
    <div style="border-bottom:1px solid #D9E2F2; padding-bottom:20px; margin-bottom:30px;
                display:flex; align-items:center;">
        {logo_html}
        <div>
            <h1 style="margin-bottom:5px;">GLP1-CardioShield</h1>
            <p style="font-size:20px; color:var(--accent-dark); font-weight:500; margin-bottom:15px;">
                Machine Learning–Assisted Cardiotoxicity Screening Platform
            </p>
            <p class="metadata">
                Computational Chemistry | Machine Learning Screening | Explainable AI
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Model Information Banner
# =====================================================

def metric_box(label: str, value: str) -> str:
    return f"""
    <div class="metric-box">
        <b>{label}</b>
        <br>
        {value}
    </div>
    """


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(metric_box("Prediction Endpoint", "hERG Ion Channel Liability"), unsafe_allow_html=True)

with col2:
    st.markdown(metric_box("Machine Learning Model", "Combined Random Forest"), unsafe_allow_html=True)

with col3:
    st.markdown(metric_box("Validation Performance", "ROC-AUC: 0.931"), unsafe_allow_html=True)

st.divider()


# =====================================================
# Molecule Input
# =====================================================

st.subheader("Compound Input")

st.markdown(
    "Enter a molecular SMILES representation to evaluate predicted hERG cardiotoxicity risk."
)

smiles = st.text_input(
    "SMILES Structure",
    value="",
    placeholder="Enter molecular SMILES...",
)

# Known reference compounds spanning a range of hERG liability, for quick testing
EXAMPLE_COMPOUNDS = {
    "Terfenadine (high liability)": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
    "Sotalol (moderate liability)": "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
    "Benchmark sulfonamide": "CCN(CC)CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "Metformin (low liability)": "CN(C)C(=N)NC(=N)N",
}

with st.expander("Example compounds"):
    for name, smi in EXAMPLE_COMPOUNDS.items():
        st.markdown(f"**{name}**")
        st.code(smi)


# =====================================================
# Risk Assessment Execution
# =====================================================

# Feature-name lookup used both in the results table and the SHAP chart
FEATURE_NAME_MAP = {
    "logP": "Lipophilicity (logP)",
    "molecular_weight": "Molecular Weight",
    "tpsa": "Polar Surface Area",
    "hbd": "Hydrogen Bond Donors",
    "hba": "Hydrogen Bond Acceptors",
    "rotatable_bonds": "Molecular Flexibility",
    "num_rings": "Ring Count",
    "aromatic_rings": "Aromatic Rings",
}


def readable_feature_name(feature: str) -> str:
    return FEATURE_NAME_MAP.get(feature, f"Morgan Fragment {feature}")


if st.button("Evaluate Compound", type="primary"):
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

        st.subheader("Compound Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <b>Predicted hERG Probability</b>
                    <h2>{risk_percent:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <b>Classification</b>
                    <h3 style="color:{risk_color};">{risk_label}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="metric-box">
                    <b>Prediction Model</b>
                    <h3>Combined RF</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ---------------------------------------------
        # Molecular Analysis
        # ---------------------------------------------

        st.subheader("Structural Analysis")

        left, right = st.columns([1, 1])

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
                unsafe_allow_html=True,
            )

            image = create_image(
                mol,
                bit_info,
                top_features
            )

            st.image(
                image,
                use_container_width=True
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
                unsafe_allow_html=True,
            )

            readable = top_features.copy()
            readable["feature"] = readable["feature"].astype(str).map(readable_feature_name)
            readable["Effect"] = readable["impact"].apply(
                lambda x: "Increases liability" if x > 0 else "Decreases liability"
            )

            table = readable[["feature", "impact", "Effect"]].rename(
                columns={"feature": "Feature", "impact": "SHAP Impact"}
            )

            st.dataframe(table, use_container_width=True, hide_index=True)

        st.divider()

        # ---------------------------------------------
        # Explainability
        # ---------------------------------------------

        st.subheader("Model Explainability")

        st.markdown(
            "SHAP values quantify how individual molecular "
            "features influenced the final model prediction."
        )

        chart_df = top_features.sort_values("impact")
        labels = [readable_feature_name(f) for f in chart_df["feature"].astype(str)]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels, chart_df["impact"], color="#0093D0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("SHAP contribution")
        ax.set_title("Top Molecular Features")

        st.pyplot(fig, use_container_width=True)

        st.divider()

        # ---------------------------------------------
        # Interpretation
        # ---------------------------------------------

        st.subheader("Scientific Interpretation")

        positive = readable[readable["impact"] > 0].head(3)
        negative = readable[readable["impact"] < 0].head(3)

        def render_interpretation_panel(title: str, items: list[str], color: str) -> None:
            bullets = "".join(f"<li>{item}</li>" for item in items)
            st.markdown(
                f"""
                <div style="background:var(--surface); border-left:4px solid {color};
                            border-radius:6px; padding:20px 22px; height:100%;
                            box-shadow:0 1px 3px rgba(20,33,43,0.06);">
                    <p style="margin:0 0 12px 0; font-weight:600; color:var(--ink);">{title}</p>
                    <ul style="margin:0; padding-left:18px; color:var(--ink-soft); line-height:1.8;">
                        {bullets}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        int_left, int_right = st.columns(2, gap="medium")

        with int_left:
            if len(positive) > 0:
                render_interpretation_panel(
                    "↑ Increases predicted liability",
                    positive["feature"].tolist(),
                    "#B3261E",
                )

        with int_right:
            if len(negative) > 0:
                render_interpretation_panel(
                    "↓ Decreases predicted liability",
                    negative["feature"].tolist(),
                    "#0093D0",
                )

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

        st.info(
            "GLP1-CardioShield provides machine learning-based estimates "
            "of potential hERG liability for research purposes. "
            "Predictions should not replace experimental validation."
        )

    except Exception as e:
        st.error(f"Unable to analyze compound: {e}")


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
    unsafe_allow_html=True,
)