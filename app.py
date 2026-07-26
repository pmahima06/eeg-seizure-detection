# 🧠 EEG Seizure Detection App
# Author: Mahima Prajapati
# Dataset: UCI Epileptic Seizure Recognition (Andrzejak et al., 2001)
# Model: Random Forest classifier trained on 178-point EEG segments

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="EEG Seizure Detection", page_icon="🧠", layout="centered")

st.title("🧠 EEG Seizure Detection")
st.caption("Built by Mahima Prajapati — binary classifier trained on the UCI Epileptic Seizure Recognition dataset")

st.markdown("""
**What this does**
- Loads a Random Forest model trained on 178-point, 1-second EEG segments
- Classifies a segment as **Seizure** or **Non-seizure** activity
- Lets you try it on real sample segments, or upload your own 178-value EEG row (CSV)

This is a learning/demonstration project using a public research dataset — not a diagnostic tool.
""")

# ============================
# Load model + scaler
# ============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("seizure_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

@st.cache_data
def load_sample_data():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=[df.columns[0]])
    return df

model, scaler = load_artifacts()
df = load_sample_data()
X_cols = [c for c in df.columns if c != "y"]

# ============================
# Try on a random sample
# ============================
st.subheader("Try it on a sample EEG segment")

col1, col2 = st.columns(2)
with col1:
    seizure_sample_btn = st.button("🔴 Load a seizure example")
with col2:
    normal_sample_btn = st.button("🔵 Load a non-seizure example")

if "current_sample" not in st.session_state:
    st.session_state.current_sample = df[df["y"] != 1].iloc[0][X_cols]

if seizure_sample_btn:
    subset = df[df["y"] == 1].sample(1, random_state=np.random.randint(0, 10000))
    st.session_state.current_sample = subset.iloc[0][X_cols]

if normal_sample_btn:
    subset = df[df["y"] != 1].sample(1, random_state=np.random.randint(0, 10000))
    st.session_state.current_sample = subset.iloc[0][X_cols]

sample = st.session_state.current_sample

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(sample.values, color="#c0392b")
ax.set_xlabel("Time step (178-sample, 1-second EEG segment)")
ax.set_ylabel("EEG amplitude")
ax.set_title("Selected EEG Segment")
st.pyplot(fig)

if st.button("🧠 Classify this segment", type="primary"):
    X_input = scaler.transform(sample.values.reshape(1, -1))
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ Predicted: **Seizure activity** (confidence: {prob:.1%})")
    else:
        st.success(f"✅ Predicted: **Non-seizure activity** (confidence: {1-prob:.1%})")

st.divider()

# ============================
# Model performance
# ============================
st.subheader("📊 Model Performance")
st.markdown("""
Trained and evaluated on an 80/20 stratified split of 11,500 EEG segments:

| Metric | Random Forest |
|---|---|
| Accuracy | 96.5% |
| Precision (seizure) | 97.0% |
| Recall (seizure) | 85.0% |
| F1 score | 0.906 |
| ROC AUC | 0.996 |

A plain logistic regression on the same raw features performed close to chance (AUC ≈ 0.50) —
the discriminative signal in raw EEG amplitude is highly non-linear, which is part of why
tree-based and deep learning models are typically used for EEG classification in practice.
""")

st.caption("Dataset source: Andrzejak et al. (2001), UCI Machine Learning Repository — Epileptic Seizure Recognition Data Set.")
