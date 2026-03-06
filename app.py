import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Model Architecture
class FraudDetector(nn.Module):
    def __init__(self):
        super(FraudDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = FraudDetector()
model.load_state_dict(torch.load("fraud_detection_model.pth"))
model.eval()

st.title("💳 Credit Card Fraud Detection")

st.write("Enter 30 transaction features")

features = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict Transaction"):

    input_data = torch.tensor([features], dtype=torch.float32)

    prediction = model(input_data)

    if prediction.item() > 0.5:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
