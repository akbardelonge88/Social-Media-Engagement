import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML Regression App",
    layout="wide"
)

# =========================
# GLOBAL CSS (LOGIN + MAIN)
# =========================
st.markdown("""
<style>

/* ===== Background ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    background-attachment: fixed;
    color: white;
}

/* ===== Hide Header ===== */
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none;}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ===== Login Card ===== */
.login-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    padding: 45px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

/* ===== Inputs ===== */
.stTextInput input, .stNumberInput input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ===== Labels ===== */
.stTextInput label, .stNumberInput label {
    color: rgba(255,255,255,0.7);
}

/* ===== Buttons ===== */
.stButton button {
    background: linear-gradient(90deg, #22d3ee, #6366f1);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px;
    font-weight: 600;
    box-shadow: 0 0 18px rgba(99,102,241,0.7);
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(34,211,238,0.95);
}

/* ===== Footer ===== */
.footer {
    position: fixed;
    bottom: 10px;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 13px;
    color: rgba(255,255,255,0.35);
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "login" not in st.session_state:
    st.session_state.login = False

# =========================
# LOGIN FUNCTION
# =========================
def login():

    st.markdown("<h1 style='text-align:center;'>🔐 HEXAMIND</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Sign in to continue</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image("login.png", width=380)

    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        user = st.text_input("User Name", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", use_container_width=True):
            if user == "admin" and pwd == "1234":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LOGIN GATE
# =========================
if not st.session_state.login:
    login()
    st.markdown("<div class='footer'>© 2026 HEXAMIND • AI & Analytics Platform</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# LOGOUT
# =========================
if st.sidebar.button("🚪 Logout"):
    st.session_state.login = False
    st.rerun()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================
# MAIN TITLE
# =========================
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
# SIDEBAR MENU
# =========================
st.sidebar.title("⚙️ Navigation")
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

    if st.button("🚀 Predict"):
        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]
        st.success(f"Predicted Value: {round(float(pred),4)}")

# =========================
# CSV UPLOAD
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

        if st.button("📈 Run Prediction"):

            preds = model.predict(df)
            df["Prediction"] = preds

            st.success("Prediction done")
            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.hist(preds, bins=20)
            ax.set_title("Distribution of Predicted Values")
            st.pyplot(fig)

            st.download_button(
                "⬇ Download Result CSV",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

# =========================
# FEATURE IMPORTANCE
# =========================
if mode == "Manual Input":

    st.header("Feature Importance")

    try:
        if hasattr(model, "named_steps"):
            final_model = list(model.named_steps.values())[-1]
        else:
            final_model = model

        if hasattr(final_model, "feature_importances_"):

            importance = final_model.feature_importances_
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
            st.pyplot(fig)

        else:
            st.info("Model does not support feature importance.")

    except Exception as e:
        st.warning("Feature importance could not be extracted.")
        st.text(e)

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>© 2026 HEXAMIND • Machine Learning Analytics System</div>", unsafe_allow_html=True)











