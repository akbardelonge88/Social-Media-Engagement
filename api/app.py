import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Engagement Prediction App", layout="wide")

st.title("📊 Engagement Prediction App")

# load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.success("Model loaded successfully")

# ===============================
# INPUT SECTION
# ===============================
st.header("Input Data")

uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview Data:", df.head())

    if st.button("Run Prediction"):
        preds = model.predict(df)

        df["Prediction"] = preds

        # kalau model classification probabilitas
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)
            df["Probability"] = prob[:,1]

        st.success("Prediction Complete")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Result", csv, "prediction.csv", "text/csv")

else:
    st.info("Upload CSV file to start prediction")