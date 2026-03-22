import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title(" Credit Card Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # -------------------------
    # BASIC DASHBOARD
    # -------------------------
    st.write("##  Data Insights")

    st.write("Shape:", data.shape)

    if "Class" in data.columns:
        fraud = data["Class"].sum()
        normal = len(data) - fraud

        st.write(f"Fraud Cases: {fraud}")
        st.write(f"Normal Cases: {normal}")

        st.write("### Class Distribution")
        st.bar_chart(data["Class"].value_counts())

    # -------------------------
    # HANDLE PREPROCESSING
    # -------------------------
    if "Time" in data.columns and "Amount" in data.columns:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        data["scaled_time"] = scaler.fit_transform(data[["Time"]])
        data["scaled_amount"] = scaler.fit_transform(data[["Amount"]])

        data = data.drop(["Time", "Amount"], axis=1)

    # Drop target if exists
    if "Class" in data.columns:
        data_model = data.drop("Class", axis=1)
    else:
        data_model = data.copy()

    # -------------------------
    # ALIGN COLUMNS
    # -------------------------
    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in data_model.columns:
            data_model[col] = 0

    data_model = data_model[expected_cols]

    # -------------------------
    # PREDICTION
    # -------------------------
    if st.button(" Run Fraud Detection"):

        predictions = model.predict(data_model)
        probabilities = model.predict_proba(data_model)[:,1]

        result = data.copy()
        result["Prediction"] = predictions
        result["Fraud Probability"] = probabilities

        result["Prediction"] = result["Prediction"].map({0: "Normal", 1: "Fraud"})

        st.write("###  Results")
        st.dataframe(result)

        # -------------------------
        # SUMMARY
        # -------------------------
        fraud_count = (predictions == 1).sum()
        total = len(predictions)

        st.write("## Summary")
        st.write(f"Fraud Detected: {fraud_count}")
        st.write(f"Fraud Percentage: {(fraud_count/total)*100:.2f}%")

        # -------------------------
        # VISUALIZATION
        # -------------------------
        st.write("## Prediction Distribution")

        fig, ax = plt.subplots()
        result["Prediction"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # Probability histogram
        st.write("## Fraud Probability Distribution")

        fig2, ax2 = plt.subplots()
        ax2.hist(probabilities, bins=30)
        st.pyplot(fig2)