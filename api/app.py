import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="ML Regression App", layout="wide")

# =========================
# SESSION LOGIN STATE
# =========================
if "login" not in st.session_state:
    st.session_state["login"] = False

# =========================
# LOGIN FUNCTION
# =========================
def login():

    st.markdown("""
        <style>

        header {visibility: hidden;}
        [data-testid="stHeader"] {display: none;}

        .stApp {
            background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
            background-attachment: fixed;
        }

        main > div {
         padding-top: 0rem;
         padding-bottom: 0rem;
        }

        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0rem;
        }    

        div[data-testid="column"] {
            display: flex;
            align-items: center;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .hero-title {
            text-align: center;
            color: white;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .hero-subtitle {
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 16px;
            margin-bottom: 35px;
        }

        .login-card {
            background: transaparent;
            backdrop-filter: blur(14px);
            padding: 45px;
            border-radius: 18px;
            border: 1px solid transparent;
            box-shadow: 0 8px 40px transparent;
        }

        .stTextInput input {
            background-color: rgba(255,255,255,0.08);
            color: black;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .stTextInput label {
            color: rgba(255,255,255,0.7);
        }

        .stButton button {
            background: linear-gradient(90deg, #22d3ee, #6366f1);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px;
            font-weight: 600;
            box-shadow: 0 0 18px rgba(99,102,241,0.7);
            transition: 0.3s;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 28px rgba(34,211,238,0.95);
        }

        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='hero-title'>Welcome Back!</div>
        <div class='hero-subtitle'>Sign in to continue to Hexamind</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
        st.image("login.png", width=380)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        user = st.text_input("User Name", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", use_container_width=True, key="login_btn"):
            if user == "admin" and pwd == "1234":
                st.session_state["login"] = True
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# COPYRIGHT LOGIN
    st.markdown("""
        <div style='text-align:center;
                    margin-top:30px;
                    font-size:12px;
                    color:rgba(255,255,255,0.5);'>
            © 2026 Hexamind. All Rights Reserved
        </div>
    """, unsafe_allow_html=True)

# =========================
# LOGOUT BUTTON
# =========================
def logout_button():
    if st.sidebar.button("🚫 Logout"):
        st.session_state["login"] = False
        st.rerun()

# =========================
# STOP IF NOT LOGIN
# =========================
if not st.session_state["login"]:
    login()
    st.stop()

logout_button()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.title("🤖 Engagement Prediction App (Regression)")

# =========================
# INSPECT PIPELINE
# =========================
def inspect_pipeline(model):

    numeric_cols = []
    categorical_cols = []
    category_map = {}

    if isinstance(model, Pipeline):
        for step_name, step in model.steps:
            if isinstance(step, ColumnTransformer):
                for name, transformer, cols in step.transformers_:

                    if isinstance(transformer, Pipeline):
                        last = transformer.steps[-1][1]
                    else:
                        last = transformer

                    if hasattr(last, "categories_"):
                        categorical_cols.extend(cols)
                        for c, cats in zip(cols, last.categories_):
                            category_map[c] = list(cats)
                    else:
                        numeric_cols.extend(cols)

    return numeric_cols, categorical_cols, category_map

numeric_cols, categorical_cols, category_map = inspect_pipeline(model)

if hasattr(model, "feature_names_in_"):
    all_cols = list(model.feature_names_in_)
else:
    all_cols = numeric_cols + categorical_cols

# =========================
# SIDEBAR MODE
# =========================
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
def validate_csv(df, all_cols, category_map):

    errors = []

    # ======================
    # STRIP SPACES HEADER
    # ======================
    df.columns = df.columns.str.strip()

    # ======================
    # CHECK KOLOM
    # ======================
    missing_cols = [c for c in all_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in all_cols]

    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if extra_cols:
        errors.append(f"Unknown columns: {extra_cols}")

    if errors:
        return errors

    # ======================
    # TRIM STRING VALUE
    # ======================
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ======================
    # CHECK NULL
    # ======================
    null_counts = df.isna().sum()
    null_cols = null_counts[null_counts > 0]

    if len(null_cols) > 0:
        errors.append(f"Null values detected in: {list(null_cols.index)}")

    # ======================
    # CHECK CATEGORY VALID
    # ======================
    for col, valid_cats in category_map.items():
        if col in df.columns:
            invalid_rows = df[~df[col].isin(valid_cats)]
            if not invalid_rows.empty:
                bad_vals = invalid_rows[col].unique().tolist()
                errors.append(f"{col} invalid values: {bad_vals[:5]}")

    # ======================
    # FORCE NUMERIC CONVERSION
    # ======================
    numeric_cols = [
        "hour","text_length","hashtag_count","mention_count",
        "impressions","month","sentiment_score","toxicity_score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                errors.append(f"{col} contains non-numeric values")

    # ======================
    # CHECK RANGE
    # ======================
    if "hour" in df.columns:
        bad = df[(df["hour"] < 0) | (df["hour"] > 24)]
        if not bad.empty:
            errors.append(f"hour out of range at rows: {bad.index.tolist()[:5]}")

    if "month" in df.columns:
        bad = df[(df["month"] < 1) | (df["month"] > 12)]
        if not bad.empty:
            errors.append(f"month out of range at rows: {bad.index.tolist()[:5]}")

    return errors

=========================
# UPLOAD CSV
# =========================
if mode == "Upload CSV":

    st.header("Upload CSV for Batch Prediction")

    template_df = pd.DataFrame(columns=all_cols)
    csv_template = template_df.to_csv(index=False)

    st.download_button(
        "⬇ Download CSV Template",
        csv_template,
        "template_input.csv",
        "text/csv"
    )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Preview", df.head())

        # ======================
        # VALIDASI CSV
        # ======================
        validation_errors = validate_csv(df, all_cols, category_map)

        if validation_errors:
            st.error("CSV validation failed ❌")
            for err in validation_errors:
                st.write(f"- {err}")
            st.stop()

        # ======================
        # PREDICT JIKA VALID
        # ======================
        if st.button("🚀 Run Prediction"):

            preds = model.predict(df)
            df["Prediction"] = preds

            st.success("Prediction done ✅")
            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.hist(preds, bins=20)
            ax.set_title("Distribution of Predicted Values")
            st.pyplot(fig)

            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )
        
# =========================
# COPYRIGHT FOOTER
# =========================
st.markdown("""
<hr style='margin-top:50px;margin-bottom:10px'>
<div style='text-align:center;
            font-size:12px;
            color:gray;'>
    © 2026 Hexamind. All Rights Reserved
</div>
""", unsafe_allow_html=True)















