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

        .main > div {
            padding-top: 1rem;
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
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            padding: 45px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 8px 40px rgba(0,0,0,0.35);
        }

        .stTextInput input {
            background-color: rgba(255,255,255,0.08);
            color: white;
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

# =========================
# LOGOUT BUTTON
# =========================
def logout_button():
    if st.sidebar.button("🚪 Logout"):
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
st.title("📊 Engagement Prediction App (Regression)")

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
if mode == "Manual Input":

    st.header("Manual Input Form")

    ordered_cols = [
        "day_of_week","hour","platform","text_length",
        "topic_category","hashtag_count","emotion_type",
        "mention_count","campaign_phase","sentiment_score",
        "impressions","toxicity_score","month"
    ]

    col_left, col_right = st.columns(2)
    input_data = {}

    for i, col in enumerate(ordered_cols):

        target_col = col_left if i % 2 == 0 else col_right

        with target_col:
            if col in category_map:
                input_data[col] = st.selectbox(col, category_map[col])
            else:
                input_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]
        st.success(f"Predicted Value: {round(float(pred),4)}")

# =========================
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

        if st.button("Run Prediction"):

            preds = model.predict(df)
            df["Prediction"] = preds

            st.success("Prediction done")
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
# FEATURE IMPORTANCE (ONLY MANUAL INPUT PAGE)
# =========================
if mode == "Manual Input":

    st.header("Feature Importance")

    try:
        # ambil model terakhir
        if hasattr(model, "named_steps"):
            final_model = list(model.named_steps.values())[-1]
        else:
            final_model = model

        if hasattr(final_model, "feature_importances_"):

            importance = final_model.feature_importances_

            # ambil nama fitur dari pipeline kalau ada
            feature_names = None

            if hasattr(model, "named_steps"):
                for step in model.named_steps.values():
                    if hasattr(step, "get_feature_names_out"):
                        feature_names = step.get_feature_names_out()
                        break

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]

            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values("Importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh(fi["Feature"], fi["Importance"])
            ax.invert_yaxis()
            ax.set_title("Top Feature Importance")
            ax.set_xlabel("Importance Score")
            plt.tight_layout()

            st.pyplot(fig)

        else:
            st.info("Model does not support feature importance.")

    except Exception as e:
        st.warning("Feature importance could not be extracted.")
        st.text(e)









