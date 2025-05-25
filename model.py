from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2048, out_features=10)

    def forward(
        self, x: torch.tensor, targets: Optional[torch.tensor] = None
    ) -> torch.tensor:
        # x: (B, 3, 32, 32)
        x = F.gelu(self.bn1(self.conv1(x)))  # (B, 96, 32, 32)
        x = F.gelu(self.bn2(self.conv2(x)))  # (B, 256, 32, 32)
        x = self.pool1(x)  # (B, 256, 16, 16)
        x = F.gelu(self.bn3(self.conv3(x)))  # (B, 256, 16, 16)
        x = F.gelu(self.bn4(self.conv4(x)))  # (B, 256, 16, 16)
        x = F.gelu(self.bn5(self.conv5(x)))  # (B, 128, 8, 8)
        x = self.pool2(x)  # (B, 128, 4, 4)
        x = x.view(x.shape[0], -1)  # (B, 128*4*4)
        x = self.dropout1(F.gelu(self.fc1(x)))  # (B, 2048)
        x = self.dropout2(F.gelu(self.fc2(x)))  # (B, 2048)
        x = self.fc3(x)  # (B, 10)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x, targets)
        return x, loss
