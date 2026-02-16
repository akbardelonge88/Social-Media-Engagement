import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="AI Engagement Predictor", layout="wide")

# ================= LOGIN SYSTEM =================
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

# logout button
col1, col2 = st.columns([8,1])
with col2:
    if st.button("Logout"):
        st.session_state["login"] = False
        st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("📊 AI Engagement Prediction Dashboard")

# ================= HELPER PIPELINE =================
def unpack_pipeline(model):
    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
        preprocessor = None
        for k in model.named_steps:
            if "preprocess" in k.lower() or "transform" in k.lower():
                preprocessor = model.named_steps[k]
        return final_model, preprocessor
    return model, None

final_model, preprocessor = unpack_pipeline(model)

# ================= TEMPLATE CSV =================
template_cols = [
    "day_of_week","platform","topic_category","emotion_type","campaign_phase",
    "followers","likes","comments","shares"
]
template_df = pd.DataFrame(columns=template_cols)

st.sidebar.download_button(
    "⬇ Download CSV Template",
    template_df.to_csv(index=False),
    "template.csv",
    "text/csv"
)

mode = st.sidebar.radio("Input Mode", ["Manual Input", "Upload CSV"])

# ================= MANUAL INPUT =================
if mode == "Manual Input":

    st.header("Manual Input")

    col1, col2 = st.columns(2)

    with col1:
        day = st.selectbox("day_of_week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        platform = st.selectbox("platform",
            ["YouTube","Twitter","Reddit","Instagram","Facebook"])

        topic = st.selectbox("topic_category",
            ["Pricing","Returns","Product","Delivery","Marketing","Support"])

    with col2:
        emotion = st.selectbox("emotion_type",
            ["Sad","Happy","Confused","Excited","Angry"])

        phase = st.selectbox("campaign_phase",
            ["Pre-Launch","Launch","Post-Launch"])

    followers = st.number_input("followers",0)
    likes = st.number_input("likes",0)
    comments = st.number_input("comments",0)
    shares = st.number_input("shares",0)

    if st.button("Predict"):

        df = pd.DataFrame([{
            "day_of_week":day,
            "platform":platform,
            "topic_category":topic,
            "emotion_type":emotion,
            "campaign_phase":phase,
            "followers":followers,
            "likes":likes,
            "comments":comments,
            "shares":shares
        }])

        pred = model.predict(df)[0]
        st.success(f"Predicted Engagement: {round(pred,3)}")

        # probability-style distribution
        preds = []
        for i in range(50):
            noise = df.copy()
            noise["likes"] += np.random.randint(-5,5)
            preds.append(model.predict(noise)[0])

        fig, ax = plt.subplots()
        ax.hist(preds, bins=20)
        ax.set_title("Prediction Distribution")
        st.pyplot(fig)

        # SHAP
        st.subheader("Explainability")
        try:
            X_proc = preprocessor.transform(df)
            explainer = shap.Explainer(final_model)
            shap_values = explainer(X_proc)

            fig = plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except:
            st.info("SHAP unavailable")

# ================= CSV MODE =================
if mode == "Upload CSV":

    st.header("Batch Prediction")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if st.button("Run Prediction"):

            preds = model.predict(df)
            df["Prediction"] = preds

            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.hist(preds, bins=20)
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

            st.download_button(
                "Download Result",
                df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

# ================= FEATURE IMPORTANCE =================
st.header("Feature Importance")

try:
    if hasattr(final_model,"feature_importances_"):
        imp = final_model.feature_importances_

        try:
            names = preprocessor.get_feature_names_out()
        except:
            names = [f"f{i}" for i in range(len(imp))]

        fi = pd.DataFrame({"Feature":names,"Importance":imp})

        fig, ax = plt.subplots(figsize=(6,4))
        fi.sort_values("Importance").plot.barh(x="Feature",y="Importance",ax=ax)
        st.pyplot(fig)

        # grouped importance
        fi["Group"] = fi["Feature"].str.split("__").str[0]
        grp = fi.groupby("Group")["Importance"].sum().sort_values(ascending=False)

        st.subheader("Grouped Importance")
        fig2, ax2 = plt.subplots()
        grp.plot.bar(ax=ax2)
        st.pyplot(fig2)

        # insight
        top = grp.index[0]
        st.success(f"📌 Biggest driver of engagement is **{top}**")

    else:
        st.info("Model has no feature importance")

except Exception as e:
    st.warning("Feature importance unavailable")