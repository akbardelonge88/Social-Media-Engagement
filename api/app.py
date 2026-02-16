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

    # background style
    st.markdown("""
        <style>
        .login-card {
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        .center-box {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 90vh;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2,1])

    # LEFT IMAGE
    with col1:
        st.image("login.png", use_container_width=True)

    # RIGHT LOGIN CARD
    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        st.markdown("## 🔐 LOGIN")

        user = st.text_input("Email")
        pwd = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
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

                    # categorical encoder
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

# fallback
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
# MANUAL INPUT (2 COLUMNS)
# =========================
if mode == "Manual Input":

    st.header("Manual Input Form")

    # urutan sesuai request lo
    ordered_cols = [
        "day_of_week","hour",
        "platform","text_length",
        "topic_category","hashtag_count",
        "emotion_type","mention_count",
        "campaign_phase","sentiment_score",
        "impressions","toxicity_score",
        "month"
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
# CSV UPLOAD + TEMPLATE
# =========================
if mode == "Upload CSV":

    st.header("Upload CSV for Batch Prediction")

    # TEMPLATE CSV
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

            # DISTRIBUTION GRAPH
            st.subheader("Prediction Distribution")

            fig, ax = plt.subplots()
            ax.hist(preds, bins=20)
            ax.set_title("Distribution of Predicted Values")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

# =========================
# FEATURE IMPORTANCE (IMPROVED)
# =========================
st.header("Feature Importance")

try:
    # ambil model terakhir
    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
    else:
        final_model = model

    if hasattr(final_model, "feature_importances_"):

        importance = final_model.feature_importances_

        # ambil nama fitur dari preprocessor
        try:
            preprocessor = model.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(20)

        # plotting lebih rapi
        fig, ax = plt.subplots(figsize=(8,6))

        bars = ax.barh(
            fi["Feature"],
            fi["Importance"]
        )

        ax.invert_yaxis()
        ax.set_title("Top Feature Importance", fontsize=14)
        ax.set_xlabel("Importance Score")

        # kasih spasi biar gak dempet
        plt.tight_layout()

        st.pyplot(fig)

    else:
        st.info("Model does not support feature importance.")

except Exception as e:
    st.warning("Feature importance could not be extracted.")
    st.text(e)