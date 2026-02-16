import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Prediction App", layout="wide")

# =========================
# SIMPLE LOGIN SYSTEM
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

st.title("📊 Engagement Prediction App")

# =========================
# AUTO DETECT FEATURES
# =========================
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    st.warning("Model has no feature_names_in_. Using manual input.")
    features = []

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.header("Manual Input Form")

    input_data = {}

    for col in features:
        input_data[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        pred = model.predict(df)[0]

        st.success(f"Prediction: {pred}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0]

            st.subheader("Probability")

            fig, ax = plt.subplots()
            ax.bar(range(len(prob)), prob)
            ax.set_title("Prediction Probability")
            st.pyplot(fig)

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

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                df["Probability"] = probs[:,1]

            st.success("Prediction done")
            st.dataframe(df)

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

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(fi["Feature"], fi["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)
else:
    st.info("Model does not support feature importance.")