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
        for _, step in model.steps:
            if isinstance(step, ColumnTransformer):
                for _, transformer, cols in step.transformers_:
                    last = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer
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
# CSV VALIDATION ENGINE
# =========================
def validate_csv(df):

    errors = []

    df.columns = df.columns.str.strip()

    # check columns
    missing = [c for c in all_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in all_cols]

    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unknown columns: {extra}")
    if errors:
        return errors

    # strip string values
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()

    # null check
    nulls = df.isna().sum()
    if (nulls > 0).any():
        errors.append(f"Null values detected in {list(nulls[nulls>0].index)}")

    # category check
    for col, cats in category_map.items():
        bad = df[~df[col].isin(cats)]
        if not bad.empty:
            errors.append(f"{col} invalid values {bad[col].unique().tolist()[:5]}")

    # numeric coercion
    numeric_cols = [
        "hour","text_length","hashtag_count","mention_count",
        "impressions","month","sentiment_score","toxicity_score"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            errors.append(f"{col} contains non numeric values")

    # range check
    if ((df["hour"]<0)|(df["hour"]>24)).any():
        errors.append("hour must be 0–24")

    if ((df["month"]<1)|(df["month"]>12)).any():
        errors.append("month must be 1–12")

    return errors

# =========================
# SIDEBAR MODE
# =========================
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input method", ["Manual Input","Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.header("Manual Input")

    ordered_cols = [
        "day_of_week","hour","platform","text_length",
        "topic_category","hashtag_count","emotion_type",
        "mention_count","campaign_phase","sentiment_score",
        "impressions","toxicity_score","month"
    ]

    col1,col2 = st.columns(2)
    input_data = {}

    for i,col in enumerate(ordered_cols):
        tgt = col1 if i%2==0 else col2
        with tgt:
            if col in category_map:
                input_data[col] = st.selectbox(col, category_map[col])
            elif col=="hour":
                input_data[col] = st.number_input(col,0,24,step=1)
            elif col=="month":
                input_data[col] = st.number_input(col,1,12,step=1)
            elif col in ["sentiment_score","toxicity_score"]:
                input_data[col] = st.number_input(col,step=0.01)
            else:
                input_data[col] = st.number_input(col,step=1)

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]
        st.success(f"Predicted Value: {round(float(pred),4)}")

# =========================
# UPLOAD CSV
# =========================
if mode=="Upload CSV":

    st.header("Upload CSV")

    template_df = pd.DataFrame(columns=all_cols)
    st.download_button("Download Template", template_df.to_csv(index=False),"template.csv")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        errors = validate_csv(df)

        if errors:
            st.error("CSV validation failed")
            for e in errors:
                st.write("-",e)
            st.stop()

        if st.button("Run Prediction"):
            preds = model.predict(df)
            df["Prediction"] = preds
            st.success("Prediction done")
            st.dataframe(df)

            fig,ax = plt.subplots()
            ax.hist(preds,bins=20)
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

            st.download_button("Download Result",df.to_csv(index=False),"prediction.csv")

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



















