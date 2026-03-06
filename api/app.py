from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from fastapi import FastAPI
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class FraudDetectionModel(nn.Module):

    def __init__(self):
        super(FraudDetectionModel, self).__init__()

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


model = FraudDetectionModel()
model.load_state_dict(torch.load("fraud_detection_model.pth"))
model.eval()


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(features: list):

    features = np.array(features)
    features = torch.tensor(features, dtype=torch.float32)

    prediction = model(features)

    if prediction.item() > 0.5:
        return {"prediction": "Fraud"}
    else:
        return {"prediction": "Normal"}
