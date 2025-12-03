import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
#   MODEL DEFINITIONS
# -----------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.float())

class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x.float())
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# -----------------------------------
#   STREAMLIT UI
# -----------------------------------

st.title(" Stock Movement Prediction App")
st.write("Predict whether a stock will **go up tomorrow** using a 20-day return window.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")

model_choice = st.selectbox("Select Model", ["MLP", "LSTM"])


if st.button("Predict"):

    try:
        # ---------------------------------------
        #   DOWNLOAD DATA
        # ---------------------------------------
        data = yf.download(ticker, period="90d", auto_adjust=False, group_by=None, progress=False)

        if data is None or data.empty:
            st.error("Ticker not found or no data available.")
            st.stop()

        # FIX MULTIINDEX (your environment returns MultiIndex ALWAYS)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)

        if "Close" not in data.columns:
            st.error("Downloaded data missing 'Close' column. Cannot proceed.")
            st.stop()

        # ---------------------------------------
        #   FEATURE GENERATION (MATCH NOTEBOOK)
        # ---------------------------------------
        data["Return"] = data["Close"].pct_change()
        data = data.dropna(subset=["Return"])

        if len(data["Return"]) < 20:
            st.error("Not enough data to compute a 20-day window.")
            st.stop()

        # Build 20-day return window
        window = data["Return"].values[-20:].astype(np.float32).reshape(1, -1)
        x = torch.tensor(window, dtype=torch.float32)

        # ---------------------------------------
        #   LOAD MODEL
        # ---------------------------------------
        if model_choice == "MLP":
            model = MLPClassifier(input_dim=20)
            weight_path = "models/saved_weights/mlp_weights.pth"

        else:  # future LSTM option
            model = LSTMClassifier()
            x = x.reshape(1, 20, 1)
            weight_path = "models/saved_weights/lstm_weights.pth"

        try:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
        except FileNotFoundError:
            st.error(f"Model weights not found at {weight_path}. Please train the model first.")
            st.stop()

        model.eval()

        # ---------------------------------------
        #   PREDICT
        # ---------------------------------------
        with torch.no_grad():
            prob = model(x).item()

        # ---------------------------------------
        #   OUTPUT
        # ---------------------------------------

        st.subheader("📉 Recent Price (Last 30 Days)")
        st.line_chart(data["Close"].tail(30))

        st.subheader("📊 Prediction Result")
        st.write(f"**Probability stock goes UP tomorrow:** `{prob:.3f}`")

        if prob >= 0.5:
            st.success("Model Suggests: **BUY** (Upward movement expected)")
        else:
            st.warning("Model Suggests: **DON'T BUY** (No upward movement expected)")

    except Exception as e:
        st.error(f"Error: {str(e)}")
