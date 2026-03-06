# 💳 Credit Card Fraud Detection using PyTorch

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project builds a **Deep Learning model using PyTorch** to detect fraudulent credit card transactions.

The system analyzes transaction features and predicts whether a transaction is **fraudulent** or **legitimate**.

Fraud detection is a critical problem in financial systems because fraudulent transactions cause billions of dollars in losses every year. This project demonstrates how machine learning can help automate fraud detection.

---

## 📊 Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Dataset source: Kaggle
https://www.kaggle.com/mlg-ulb/creditcardfraud

### Dataset Characteristics

* Transactions made by European cardholders
* **284,807 total transactions**
* **492 fraud cases**
* Highly **imbalanced dataset**
* Features are anonymized using **PCA transformation**

### Feature Description

| Feature | Description                                 |
| ------- | ------------------------------------------- |
| V1–V28  | PCA anonymized features                     |
| Time    | Seconds elapsed between transactions        |
| Amount  | Transaction amount                          |
| Class   | Target variable (0 = Legitimate, 1 = Fraud) |

---

## ⚙️ Project Workflow

This project follows a complete **Machine Learning pipeline**.

### 1️⃣ Data Loading

* Dataset loaded using **Pandas**
* Initial inspection of dataset structure

### 2️⃣ Exploratory Data Analysis (EDA)

Performed analysis such as:

* Fraud vs legitimate transaction distribution
* Transaction amount distribution
* Statistical summary of dataset
* Visualization using **Matplotlib** and **Seaborn**

### 3️⃣ Data Preprocessing

Steps performed:

* Feature scaling using **StandardScaler**
* Handling class imbalance
* Train–Test split (80/20)

---

## 🧠 Neural Network Model (PyTorch)

A **Feedforward Neural Network** was implemented using PyTorch.

### Model Architecture

Input Layer
→ 30 Features

Hidden Layers
→ Dense Layer
→ ReLU Activation

Hidden Layers
→ Dense Layer
→ ReLU Activation

Output Layer
→ 1 Neuron
→ Sigmoid Activation

### Model Flow

Input Layer (30 features)
↓
Linear Layer
↓
ReLU Activation
↓
Linear Layer
↓
ReLU Activation
↓
Output Layer
↓
Sigmoid Activation

---

## 🏋️ Model Training

Training configuration:

* **Loss Function:** Binary Cross Entropy Loss
* **Optimizer:** Adam
* **Epochs:** 10–50

Training process:

1. Forward pass
2. Loss computation
3. Backpropagation
4. Weight update using optimizer

---

## 📈 Model Evaluation

The trained model was evaluated using multiple performance metrics.

### Evaluation Metrics

* **Classification Report**
* **Confusion Matrix**
* **ROC-AUC Score**

| Metric    | Purpose                                |
| --------- | -------------------------------------- |
| Precision | Accuracy of fraud predictions          |
| Recall    | Ability to detect fraud cases          |
| F1-score  | Balance between precision and recall   |
| ROC-AUC   | Model's ability to distinguish classes |

Visualization libraries used:

* Matplotlib
* Seaborn

---

## 💾 Model Saving

After training, the model is saved using PyTorch so it can be reused without retraining.

Example:

```python
torch.save(model.state_dict(), "fraud_detection_model.pth")
```

---

## 🚀 Web App Deployment using Streamlit

An interactive web application was built using **Streamlit** to allow users to test the fraud detection model in real time.

Users can enter transaction feature values and instantly receive a prediction indicating whether the transaction is **fraudulent or legitimate**.

The dashboard also displays a **fraud risk score** to visualize the model's prediction confidence.

---

## ▶ Running the Streamlit App

Install dependencies:

pip install -r requirements.txt

Run the Streamlit dashboard:

streamlit run app.py

The application will open automatically in your browser:

http://localhost:8501


## 🛠 Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Programming language         |
| PyTorch      | Deep learning framework      |
| Pandas       | Data processing              |
| Scikit-learn | Preprocessing and evaluation |
| Streamlit    | Interactive web dashboard    |
| Matplotlib   | Data visualization           |
| Seaborn      | Statistical visualization    |

---

## 📂 Project Structure

```
ML_pytorch-credit-fraud-detection
│
├── notebooks
│   └── training.ipynb
│
├── app.py
│
├── fraud_detection_model.pth
├── requirements.txt
└── README.md
```
## 🌐 Live Demo

Try the application here:

https://ml-pytorch-credit-fraud-detection.streamlit.app/

---

## 🎯 Future Improvements

* Handle dataset imbalance using **SMOTE**
* Try **Autoencoder-based anomaly detection**
* Deploy using **Docker**
* Create a **web dashboard for fraud monitoring**

---

## 👨‍💻 Author

Developed as a **Deep Learning project for Credit Card Fraud Detection using PyTorch**.
