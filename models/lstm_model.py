import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
