import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
# LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------
# HERO SECTION
# -------------------------
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
Detect fraudulent transactions using machine learning.  
This system analyzes transaction patterns and identifies high-risk activities.

**Key Features:**
- Real-time fraud prediction
- Probability-based risk scoring
- Data insights & visualization
""")

st.divider()

# -------------------------
# MODEL INFO
# -------------------------
with st.expander(" How the Model Works"):
    st.write("""
- Trained on highly imbalanced dataset  
- Uses feature-transformed inputs (V1–V28)  
- Outputs probability of fraud  
- Optimized for detecting rare fraudulent transactions  
""")

# -------------------------
# DATA INSIGHTS SECTION
# -------------------------
st.header("📊 Dataset Insights")

sample_info = {
    "Total Transactions": "284,807",
    "Fraud Cases": "492",
    "Fraud Rate": "0.17%"
}

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", sample_info["Total Transactions"])
col2.metric("Fraud Cases", sample_info["Fraud Cases"])
col3.metric("Fraud Rate", sample_info["Fraud Rate"])

# st.info(" Fraud detection is a highly imbalanced problem. Even small improvements matter.")

st.divider()

# -------------------------
# UPLOAD SECTION (MOVED LAST)
# -------------------------
st.header("📂 Upload Dataset for Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV file (same format as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.success(" Dataset uploaded successfully")

    # Preview (inside expander to reduce clutter)
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(data.head())

    # -------------------------
    # BASIC INSIGHTS
    # -------------------------
    if "Class" in data.columns:
        fraud = int(data["Class"].sum())
        total = len(data)
        normal = total - fraud

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Fraud Cases", fraud)
        col3.metric("Fraud Rate", f"{(fraud/total)*100:.2f}%")

        # Class Distribution
        st.subheader("Class Distribution")
        st.bar_chart(data["Class"].value_counts())

    # -------------------------
    # PREPROCESSING
    # -------------------------
    if "Time" in data.columns and "Amount" in data.columns:
        scaler = StandardScaler()
        data["scaled_time"] = scaler.fit_transform(data[["Time"]])
        data["scaled_amount"] = scaler.fit_transform(data[["Amount"]])
        data = data.drop(["Time", "Amount"], axis=1)

    # Prepare model input
    if "Class" in data.columns:
        data_model = data.drop("Class", axis=1)
    else:
        data_model = data.copy()

    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in data_model.columns:
            data_model[col] = 0

    data_model = data_model[expected_cols]

    st.divider()


    st.subheader(" Model Sensitivity Control")

    threshold = st.slider(
        "Adjust Fraud Detection Threshold",
        0.0, 1.0, 0.5, 0.01
    )
    # -------------------------
    # PREDICTION BUTTON
    # -------------------------
    if st.button(" Run Fraud Detection"):

        with st.spinner("Analyzing transactions..."):
            probabilities = model.predict_proba(data_model)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

        result = data.copy()
        result["Prediction"] = predictions
        result["Fraud Probability"] = probabilities
        result["Prediction"] = result["Prediction"].map({0: "Normal", 1: "Fraud"})

        # -------------------------
        # SUMMARY
        # -------------------------
        fraud_count = int((predictions == 1).sum())
        total = len(predictions)
        fraud_percent = (fraud_count / total) * 100

        st.header("📌 Results Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", total)
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Fraud %", f"{fraud_percent:.2f}%")

        # Insight message
        if fraud_percent < 1:
            st.success("Low fraud risk detected in dataset")
        elif fraud_percent < 5:
            st.warning("Moderate fraud risk detected")
        else:
            st.error("High fraud risk detected!")

        st.divider()
        # -------------------------
            # MODEL INTERPRETATION (ADDED)
            # -------------------------
        st.subheader("🧠 Model Interpretation")

        st.write(f"""
            Out of {total} transactions:

            - Model flagged **{fraud_count} transactions as fraud**
            - Risk level: **{fraud_percent:.2f}%**

             Model may predict more fraud to reduce missed cases.
            """)
                    
        # -------------------------
        #  Confusion Matrix
        # -------------------------
        if "Class" in data.columns:

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(data["Class"], predictions)

            fig_cm, ax_cm = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)

            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

            st.pyplot(fig_cm)

        # -------------------------
        # ROC Curve
        # -------------------------
        if "Class" in data.columns:

            fpr, tpr, _ = roc_curve(data["Class"], probabilities)
            roc_auc = auc(fpr, tpr)

            st.subheader("ROC Curve")

            fig_roc, ax_roc = plt.subplots(figsize=(4,3))
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")

            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()

            st.pyplot(fig_roc)

        
        # -------------------------
        # Precision-Recall Curve
        # -------------------------
        if "Class" in data.columns:

            precision, recall, _ = precision_recall_curve(data["Class"], probabilities)

            st.subheader("Precision-Recall Curve")

            fig_pr, ax_pr = plt.subplots(figsize=(4,3))
            ax_pr.plot(recall, precision)

            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")

            st.pyplot(fig_pr)

            
        # -------------------------
        # VISUALIZATIONS
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Distribution")
            fig, ax = plt.subplots(figsize=(4,3))
            result["Prediction"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Fraud Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.hist(probabilities, bins=30)
            st.pyplot(fig2)

        # -------------------------
        # RESULT TABLE (OPTIONAL)
        # -------------------------
        with st.expander(" View Detailed Results"):
            st.dataframe(result)