import torch
import torch.nn as nn

class CreditNet(nn.Module):
    def __init__(self, input_size=10):
        super(CreditNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)