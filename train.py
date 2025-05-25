import random
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from async_logger import AsyncLogger
from model import CNN

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2023, 0.1994, 0.2010]
augment = transforms.Compose(
    [
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ]
)
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=augment
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
)

B = 1024
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

model = CNN()
print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model.to(device)
model.compile()

optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
val_cadence = 100


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


logger = AsyncLogger(path=f"metrics-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jsonl")

for step in range(1000):
    if step % val_cadence == 0:
        model.eval()
        with torch.no_grad():
            total_samples = 0
            val_loss = 0.0
            val_correct = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                batch_size = x.shape[0]
                val_loss += loss.item() * batch_size
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == y).sum().item()
                total_samples += batch_size
            val_loss /= total_samples
            val_acc = val_correct / total_samples
            logger(
                step=step,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            print(f"step {step:>5} | val_loss {val_loss:.6f} | val_acc {val_acc:.4f}")
        model.train()

    optim.zero_grad()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss.backward()
    optim.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    grad_norm = get_grad_norm(model)
    elapsed = (end - start) * 1e3
    acc = (torch.argmax(logits, dim=1) == y).float().mean()

    logger(
        step=step,
        train_loss=loss.item(),
        train_acc=acc.item(),
        grad_norm=grad_norm,
        step_time=elapsed,
    )
    print(
        f"step {step:>5} | train_loss {loss.item():.6f} | train_acc {acc.item():.4f} | grad norm {grad_norm:.4f} | step time {elapsed:.2f}ms"
    )

logger.close()
