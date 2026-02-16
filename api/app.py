import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="ML Regression App", layout="wide")

# =========================
# LOGIN
# =========================
def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state["login"] = True
        else:
            st.error("Invalid credentials")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.title("📊 Engagement Prediction App (Regression)")

# =========================
# AUTO DETECT PIPELINE STRUCTURE
# =========================
def inspect_pipeline(model):

    numeric_cols = []
    categorical_cols = []
    category_map = {}

    if isinstance(model, Pipeline):

        for step_name, step in model.steps:

            if isinstance(step, ColumnTransformer):

                for name, transformer, cols in step.transformers_:

                    # numeric
                    if "num" in name.lower():
                        numeric_cols.extend(cols)

                    # categorical encoder
                    if hasattr(transformer, "categories_"):
                        categorical_cols.extend(cols)

                        for c, cats in zip(cols, transformer.categories_):
                            category_map[c] = list(cats)

                    # pipeline inside columntransformer
                    if isinstance(transformer, Pipeline):
                        last = transformer.steps[-1][1]

                        if hasattr(last, "categories_"):
                            categorical_cols.extend(cols)

                            for c, cats in zip(cols, last.categories_):
                                category_map[c] = list(cats)

                        else:
                            numeric_cols.extend(cols)

    return numeric_cols, categorical_cols, category_map


numeric_cols, categorical_cols, category_map = inspect_pipeline(model)

# fallback kalau model bukan pipeline
if hasattr(model, "feature_names_in_"):
    all_cols = list(model.feature_names_in_)
else:
    all_cols = numeric_cols + categorical_cols

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.header("Manual Input Form")
    input_data = {}

    for col in all_cols:

        # categorical auto from encoder
        if col in category_map:
            input_data[col] = st.selectbox(col, category_map[col])

        # numeric auto
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
    file = st.file_uploader("Upload file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Preview", df.head())

        if st.button("Run Prediction"):

            preds = model.predict(df)
            df["Prediction"] = preds

            st.success("Prediction done")
            st.dataframe(df)

            # =========================
            # DISTRIBUTION GRAPH
            # =========================
            st.subheader("Prediction Distribution")

            fig, ax = plt.subplots()
            ax.hist(preds, bins=20)
            ax.set_title("Distribution of Predicted Values")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.download_button(
                "Download Result",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

# =========================
# FEATURE IMPORTANCE
# =========================
st.header("Feature Importance")

try:
    final_model = model[-1] if isinstance(model, Pipeline) else model

    if hasattr(final_model, "feature_importances_"):

        fi = pd.DataFrame({
            "Feature": getattr(model, "feature_names_in_", range(len(final_model.feature_importances_))),
            "Importance": final_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(fi["Feature"], fi["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

    else:
        st.info("Model does not expose feature_importances_")

except:
    st.info("Feature importance unavailable")