import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ================= USER LOGIN =================
USERS = {
    "akbar": {"password": "123", "role": "premium"},
    "guest": {"password": "123", "role": "free"},
}

if "user" not in st.session_state:
    st.session_state.user = None

def login_page():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in USERS and USERS[u]["password"] == p:
            st.session_state.user = {"name": u, "role": USERS[u]["role"]}
            st.rerun()
        else:
            st.error("Login gagal")

def logout_btn():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.rerun()

if st.session_state.user is None:
    login_page()
    st.stop()

logout_btn()

st.sidebar.success(f"Login: {st.session_state.user['name']}")
st.sidebar.info(f"Role: {st.session_state.user['role']}")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("model.pkl","rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    st.warning("Model belum ditemukan (model.pkl)")
    model = None

# ================= TEMPLATE CSV =================
template = pd.DataFrame({
    "feature_1":[1],
    "feature_2":[2],
    "category":["A"]
})

csv_template = template.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Template CSV",
    csv_template,
    "template.csv",
    "text/csv"
)

# ================= UPLOAD DATA =================
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    st.write("Preview data", df.head())

    # ===== AUTO DETECT TYPE =====
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.write("Numeric:", numeric_cols)
    st.write("Categorical:", cat_cols)

    # ===== ENCODING SIMPLE =====
    df_encoded = df.copy()
    encoders = {}

    for c in cat_cols:
        df_encoded[c] = df_encoded[c].astype("category")
        encoders[c] = dict(enumerate(df_encoded[c].cat.categories))
        df_encoded[c] = df_encoded[c].cat.codes

    # ===== PREDICT =====
    if model:
        preds = model.predict(df_encoded)
        df["prediction"] = preds
        st.success("Prediksi selesai")
        st.write(df.head())

        # ===== DISTRIBUTION PLOT =====
        st.subheader("Distribusi Prediksi")
        fig, ax = plt.subplots()
        ax.hist(preds, bins=20)
        st.pyplot(fig)

        # ===== FEATURE IMPORTANCE =====
        st.subheader("Feature Importance")

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(
                model.feature_importances_,
                index=df_encoded.columns
            ).sort_values(ascending=False)

            # group by category name before underscore
            grouped = imp.groupby(lambda x: x.split("_")[0]).sum()

            st.bar_chart(grouped)

            top_feature = grouped.idxmax()
            st.success(f"🔥 Driver utama prediksi: {top_feature}")

            st.info(f"📊 Insight: Variabel '{top_feature}' paling berpengaruh terhadap hasil model.")

        else:
            st.warning("Model tidak punya feature importance")

        # ===== SHAP PREMIUM ONLY =====
        if st.session_state.user["role"] != "premium":
            st.warning("🔒 SHAP hanya untuk premium")
        else:
            st.subheader("SHAP Explainability")

            try:
                import shap
                explainer = shap.Explainer(model, df_encoded)
                shap_values = explainer(df_encoded[:50])

                fig2 = plt.figure()
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig2)

            except Exception as e:
                st.error("SHAP gagal dijalankan. Install shap dulu.")