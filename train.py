import random
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import MLP

device = torch.device('cuda:0')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2023, 0.1994, 0.2010]
augment = transforms.Compose([
    transforms.RandomCrop(size=(32, 32), padding=4), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augment)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(seed))

B = 1024
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

model = MLP()
print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
val_cadence = 100

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

for step in range(1000):
    if step % val_cadence == 0:
        model.eval()
        with torch.no_grad():
            total_samples = 0
            val_loss = 0.0 
            val_correct = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                batch_size = x.shape[0]
                val_loss += loss.item() * batch_size
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == y).sum().item()
                total_samples += batch_size
            val_loss /= total_samples
            val_acc = val_correct / total_samples
            print(f'val loss {val_loss:.6f} | val acc {val_acc:.4f}')
        model.train()

    optim.zero_grad()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    start = time.perf_counter()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss.backward()
    optim.step()
    torch.cuda.synchronize()
    end = time.perf_counter()

    pred = torch.argmax(logits, dim=1)
    acc = (pred == y).float().mean()
    print(f'step {step:>5} | loss {loss.item():.6f} | acc {acc.item():.4f} | grad norm {get_grad_norm(model):.4f} | time {(end - start)*1e3:.2f}ms')
