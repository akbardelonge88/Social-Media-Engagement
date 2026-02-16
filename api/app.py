import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Hexamind ML App", layout="wide")

# =========================
# SESSION STATE
# =========================
if "login" not in st.session_state:
    st.session_state.login = False

# =========================
# LOGIN UI
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

        /* CENTER VERTICAL */
        .center-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 75vh;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .hero-title {
            text-align: left;
            color: white;
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .hero-subtitle {
            text-align: left;
            color: rgba(255,255,255,0.6);
            font-size: 15px;
            margin-bottom: 25px;
        }

        .login-card {
            background: rgba(255, 255, 255, 0.06);
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

        footer {visibility: hidden;}

        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("<div class='center-wrapper'>", unsafe_allow_html=True)
        st.image("login.png", width=360)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='center-wrapper'>", unsafe_allow_html=True)

        st.markdown("""
            <div>
                <div class='hero-title'>Welcome Back!</div>
                <div class='hero-subtitle'>
                    Sign in to continue to Hexamind
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        user = st.text_input("User Name", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", use_container_width=True, key="login_btn"):
            if user == "admin" and pwd == "1234":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STOP IF NOT LOGIN
# =========================
if not st.session_state.login:
    login()
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.success("✅ Logged in")
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

st.title("📊 Hexamind Engagement Prediction")

# =========================
# INSPECT PIPELINE
# =========================
def inspect_pipeline(model):

    numeric_cols = []
    categorical_cols = []
    category_map = {}

    if isinstance(model, Pipeline):
        for _, step in model.steps:
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
# INPUT MODE
# =========================
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.header("Manual Input Form")

    ordered_cols = all_cols
    col_left, col_right = st.columns(2)
    input_data = {}

    for i, col in enumerate(ordered_cols):
        target = col_left if i % 2 == 0 else col_right

        with target:
            if col in category_map:
                input_data[col] = st.selectbox(col, category_map[col])
            else:
                input_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]
        st.success(f"Predicted Value: {round(float(pred),4)}")

# =========================
# CSV UPLOAD
# =========================
if mode == "Upload CSV":

    st.header("Upload CSV for Batch Prediction")

    template_df = pd.DataFrame(columns=all_cols)
    st.download_button(
        "⬇ Download CSV Template",
        template_df.to_csv(index=False),
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
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

# =========================
# FOOTER
# =========================
st.markdown("""
    <hr style="margin-top:50px;">
    <div style='text-align:center; color:gray; font-size:13px;'>
        © 2026 Hexamind AI · Machine Learning Analytics Platform
    </div>
""", unsafe_allow_html=True)


















