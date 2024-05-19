# Example model used in CamoTSS

import torch
import torch.nn as nn

class CNN1d_base(nn.Module):
    def __init__(self):
        super(CNN1d_base, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, 8),
            nn.ReLU(),
            nn.Conv1d(128, 64, 4),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(3040, 32),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
