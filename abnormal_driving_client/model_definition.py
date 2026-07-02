from __future__ import annotations

import torch
import torch.nn as nn


WINDOW_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "海拔",
    "vss速度",
    "空档信号",
    "喇叭信号",
    "倒挡信号",
    "制动信号",
    "左转向灯信号",
    "右转向灯信号",
    "远光灯信号",
    "近光灯信号",
    "ACC状态",
    "与正北方向夹角",
]


class CNNLSTMModel(nn.Module):
    """CNN-BiLSTM model used by the prototype client."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)
