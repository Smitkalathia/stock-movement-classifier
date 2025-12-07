#  Stock Movement Prediction Using Neural Networks  
CSCI 4050U — Machine Learning Final Project

##  Overview  
This project predicts **next-day stock movement** (UP or DOWN) using the past **20 days of daily returns**.  
Two neural network architectures were trained:

- **Multilayer Perceptron (MLP)**  
- **Long Short-Term Memory Network (LSTM)**  

The deployed system uses the **MLP model** inside a Streamlit web application to make real-time predictions using live market data.

---

##  Dataset  
Historical stock price data was collected through **Yahoo Finance (yfinance)**.

### Tickers used:
All S&P 500 stocks

For each stock, we retrieved daily OHLCV values and computed:

### Daily Return:
Return_t = (Close_t - Close_(t−1)) / Close_(t−1)

### Target Label:
- `1` → next-day return > 0  
- `0` → next-day return ≤ 0  

Data was cleaned, merged, and used to create 20-day sliding windows for training.

---

##  Model Architectures

### **1. Multilayer Perceptron (MLP)**  
Input dimension: **20 (20-day window of returns)**

Architecture:
Linear(20 → 128)
ReLU
Dropout(0.2)
Linear(128 → 64)
ReLU
Linear(64 → 1)
Sigmoid → Probability of next-day price increase

---

### **2. Long Short-Term Memory (LSTM)**  
Input shape: `(batch_size, 20, 1)`

Architecture:
LSTM(input_size=1 → hidden_size=64)
Linear(64 → 1)
Sigmoid → Probability of next-day increase

---

##  Training Pipeline

The training notebook performs:

1. Downloading and cleaning stock data  
2. Computing daily returns  
3. Building 20-day sliding windows  
4. Splitting into Train / Validation / Test sets  
5. Training both MLP and LSTM  
6. Evaluating model accuracy  
7. Saving model weights for deployment  

### Training notebook:
notebooks/training.ipynb

### Final Accuracy (representative runs):
- **MLP:** ~0.545 
- **LSTM:** ~0.53  

These values are realistic because short-term stock movement is highly noisy and near-random.

---

##  Deployment (Streamlit App)

A Streamlit-based web app deploys the trained MLP model.

### Run the app locally:

```bash
python -m streamlit run deployment/app.py
```
##  App Features

- Fetches recent OHLCV data for any ticker  
- Computes a 20-day return window  
- Loads the trained MLP model  
- Outputs:
  - Probability the stock will go **UP** tomorrow  
  - **BUY / DON'T BUY** recommendation  
- Displays a graph of recent price trends  

---

##  Repository Structure

 stock_project
├── deployment/
│ └── app.py # Streamlit application
│
├── models/
│ └── saved_weights/
│ ├── mlp_weights.pth # Trained MLP model weights
│ └── lstm_weights.pth # Optional: trained LSTM weights
│
├── notebooks/
│ └── training.ipynb # Full training notebook
│
├── requirements.txt # Project dependencies
└── README.md # Documentation

---

##  Deliverables

This project includes:

- GitHub Repository (public)  
- Training Notebook (Jupyter)  
- Trained Model Weights  
- Streamlit Demo Application  
- Presentation Slides (PDF)  
- YouTube Demo Video (max 10 minutes)  

links:
YouTube Demo: <insert link>
Slides (PDF): <insert link>


---

##  Team

- **Smit Kalathia**  
- **Samir Ahmadi**
- **Kyle Liao**

---

##  Disclaimer

This project is for **academic purposes only**.  
Stock predictions are inherently uncertain and should **not** be used for real financial decisions.

---

##  Summary

This project demonstrates a complete machine learning workflow:

- Data acquisition  
- Feature engineering  
- Neural network training  
- Model evaluation  
- Deployment through a web interface  

It provides a practical example of integrating ML models into real-world applications.

