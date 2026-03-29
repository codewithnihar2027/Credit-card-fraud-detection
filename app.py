import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# LOAD MODEL (SAFE)
# -------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# -------------------------
# HERO SECTION
# -------------------------
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
Detect fraudulent transactions using machine learning.  
This system analyzes transaction patterns and identifies high-risk activities.
""")

st.divider()

# -------------------------
# DATA INSIGHTS
# -------------------------
st.header("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Rate", "0.17%")

st.divider()

# -------------------------
# FILE UPLOAD
# -------------------------
st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    #  MEMORY FIX
    if len(data) > 50000:
        data = data.sample(50000)

    st.success("Dataset uploaded")

    st.dataframe(data.head())

    # -------------------------
    # BASIC INFO
    # -------------------------
    if "Class" in data.columns:
        fraud = int(data["Class"].sum())
        total = len(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Fraud Cases", fraud)
        col3.metric("Fraud Rate", f"{(fraud/total)*100:.2f}%")

        st.bar_chart(data["Class"].value_counts())

    # -------------------------
    # PREPROCESSING
    # -------------------------
    if "Time" in data.columns and "Amount" in data.columns:
        scaler = StandardScaler()
        data["scaled_time"] = scaler.fit_transform(data[["Time"]])
        data["scaled_amount"] = scaler.fit_transform(data[["Amount"]])
        data = data.drop(["Time", "Amount"], axis=1)

    if "Class" in data.columns:
        data_model = data.drop("Class", axis=1)
    else:
        data_model = data.copy()

    # Match columns
    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in data_model.columns:
            data_model[col] = 0

    data_model = data_model[expected_cols]

    # -------------------------
    # THRESHOLD
    # -------------------------
    threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5)

    # -------------------------
    # PREDICT
    # -------------------------
    if st.button("Run Detection"):

        with st.spinner("Processing..."):
            probabilities = model.predict_proba(data_model)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

        result = data.copy()
        result["Prediction"] = predictions
        result["Fraud Probability"] = probabilities

        fraud_count = int((predictions == 1).sum())
        total = len(predictions)

        st.header("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Fraud", fraud_count)
        col3.metric("Fraud %", f"{(fraud_count/total)*100:.2f}%")

        # -------------------------
        # CONFUSION MATRIX (NO SEABORN)
        # -------------------------
        if "Class" in data.columns:
            cm = confusion_matrix(data["Class"], predictions)

            fig, ax = plt.subplots()
            ax.imshow(cm)

            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, cm[i][j], ha="center", va="center")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

        # -------------------------
        # ROC
        # -------------------------
        if "Class" in data.columns:
            fpr, tpr, _ = roc_curve(data["Class"], probabilities)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], "--")

            st.pyplot(fig)

        # -------------------------
        # RESULT TABLE
        # -------------------------
        with st.expander("View Data"):
            st.dataframe(result)