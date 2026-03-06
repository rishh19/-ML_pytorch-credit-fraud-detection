import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Page settings
st.set_page_config(page_title="Fraud Detection AI", layout="wide")

# Model architecture
class FraudDetector(nn.Module):
    def __init__(self):
        super(FraudDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(30,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.network(x)

# Load model
model = FraudDetector()
model.load_state_dict(torch.load("fraud_detection_model.pth", map_location="cpu"))
model.eval()

# Sidebar
st.sidebar.title("Fraud AI Dashboard")
st.sidebar.write("Credit Card Fraud Detection System")

# Title
st.title("💳 Credit Card Fraud Detection")

st.write(
"""
This system predicts whether a transaction is **fraudulent or legitimate** using a trained
**PyTorch neural network model**.
"""
)

st.divider()

# Layout
col1, col2 = st.columns(2)

features = []

with col1:
    st.subheader("Transaction Features")

    for i in range(15):
        value = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(value)

with col2:
    st.subheader("Additional Features")

    for i in range(15,30):
        value = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(value)

st.divider()

# Predict Button
if st.button("Run Fraud Detection", use_container_width=True):

    input_tensor = torch.tensor([features], dtype=torch.float32)

    prediction = model(input_tensor).item()

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error(f"Fraudulent Transaction Detected\n\nFraud Probability: {prediction:.2f}")
    else:
        st.success(f"Legitimate Transaction\n\nFraud Probability: {prediction:.2f}")

    st.progress(prediction)
