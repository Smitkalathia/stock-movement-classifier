# Stock Movement Prediction Using Neural Networks  
CSCI 4050U — Machine Learning Final Project

---

## Overview  
This project predicts **next-day stock movement** (UP or DOWN) using the past **20 days of daily returns**.  
Two neural network architectures were trained:

- **Multilayer Perceptron (MLP)**  
- **Long Short-Term Memory (LSTM)**  

The deployed system uses the **MLP model** inside a Streamlit web application to make real-time predictions using live market data.

---

## Dataset  
Historical stock price data was collected using **Yahoo Finance (yfinance)**.

### Tickers used:
All **S&P 500 companies**.

### Daily Return:
```
Return_t = (Close_t − Close_(t−1)) / Close_(t−1)
```

### Target Label:
- `1` → next-day return > 0  
- `0` → next-day return ≤ 0  

After cleaning and merging data, 20-day sliding windows were generated for training.

---

## Model Architectures

### 1. Multilayer Perceptron (MLP)

**Input:** 20-day window of returns  

Architecture:
```
Linear(20 → 128)
ReLU
Dropout(0.2)
Linear(128 → 64)
ReLU
Linear(64 → 1)
Sigmoid  → probability of next-day price increase
```

---

### 2. Long Short-Term Memory (LSTM)

**Input:** (batch_size, 20, 1)

Architecture:
```
LSTM(1 → hidden_size=64)
Linear(64 → 1)
Sigmoid → probability of next-day increase
```

---

## Training Pipeline

The notebook performs:

1. Downloading and cleaning stock data  
2. Calculating daily returns  
3. Creating 20-day sliding windows  
4. Splitting Train / Validation / Test  
5. Training MLP and LSTM models  
6. Evaluating accuracy  
7. Saving model weights for deployment  

Training located here:
```
notebooks/training.ipynb
```

### Accuracy (Typical Results)
- **MLP:** ~0.545  
- **LSTM:** ~0.53  

These are realistic because short-term stock movements are nearly random.

---

## How to Run the Application

Follow the steps below to run the Streamlit stock prediction app locally.

---

### 1. Clone the Repository
```bash
git clone https://github.com/Smitkalathia/stock-movement-classifier.git
cd stock-movement-classifier
```

---

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Run the Streamlit App
```bash
python -m streamlit run deployment/app.py
```

This will open:
```
http://localhost:8501
```

---

## App Features

- Fetches recent OHLCV data for any ticker  
- Computes a 20-day return window  
- Loads the trained MLP model  
- Outputs:
  - Probability the stock will go **UP** tomorrow  
  - **BUY / DON'T BUY** recommendation  
- Displays recent price graph  

---

## Repository Structure

```
stock_project
├── deployment/
│   └── app.py                     # Streamlit application
│
├── models/
│   └── saved_weights/
│       ├── mlp_weights.pth        # Trained MLP model weights
│       └── lstm_weights.pth       # Optional: LSTM model weights
│
├── notebooks/
│   └── training.ipynb             # Full training notebook
│
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## Deliverables

- Public GitHub Repository  
- Jupyter Training Notebook  
- Trained Model Weights  
- Streamlit Demo Application  
- Presentation Slides (PDF)  
- YouTube Demo Video (≤ 10 minutes)

Links:
- YouTube Demo: `<insert link>`  
- Slides (PDF): `<insert link>`  

---

## Team

- **Smit Kalathia**  
- **Samir Ahmadi**  
- **Kyle Liao**

---

## Disclaimer
This project is for **academic purposes only**.  
Stock predictions are uncertain and should **not** be used for real-world financial decisions.

---

## Summary

This project demonstrates a full ML workflow:

- Data acquisition  
- Feature engineering  
- Neural network training  
- Evaluation and model selection  
- Deployment using Streamlit  

It provides a real example of integrating trained neural models into a usable application.

