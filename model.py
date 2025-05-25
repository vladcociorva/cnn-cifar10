from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x: torch.tensor, targets: Optional[torch.tensor] = None) -> torch.tensor: 
        # x = (B, 3, 32, 32)
        x = x.view(x.shape[0], -1) # flatten to (B, 3*32*32)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        loss = None
        if targets is not None: 
            loss = F.cross_entropy(x, targets)
        return x, loss
