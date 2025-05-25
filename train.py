import random
import time
import os
import numpy as np
import torch
import torchvision
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from async_logger import AsyncLogger
from model import CNN
from utils import get_grad_norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--metrics-dir", type=str, default="metrics", help="Directory to save metrics"
    )
    parser.add_argument(
        "--val-cadence", type=int, default=5, help="Validation cadence in epochs"
    )
    parser.add_argument(
        "--checkpoint-cadence",
        type=int,
        default=50,
        help="Checkpoint cadence in epochs",
    )
    return parser.parse_args()


args = parse_args()

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
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

B = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
steps_per_epoch = len(train_loader)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

model = CNN()
print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model.to(device)
model.compile()

optim = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, "min", factor=0.1, patience=10
)
val_cadence = args.val_cadence
checkpoint_cadence = args.checkpoint_cadence
checkpoint_dir = args.checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(args.metrics_dir, exist_ok=True)

logger = AsyncLogger(
    path=f"{args.metrics_dir}/metrics-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jsonl"
)

for epoch in range(args.epochs):
    if epoch % val_cadence == 0:
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
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            print(f"epoch {epoch:>3} | val_loss {val_loss:.6f} | val_acc {val_acc:.4f}")
        model.train()
        scheduler.step(val_loss)

    if epoch != 0 and epoch % checkpoint_cadence == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    for step, (x, y) in enumerate(train_loader, start=epoch * steps_per_epoch):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()

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
            lr=optim.param_groups[0]["lr"],
        )
        print(
            f"epoch {epoch:>3} | step {step:>5} | train_loss {loss.item():.6f} | train_acc {acc.item():.4f} | grad norm {grad_norm:.4f} | step time {elapsed:.2f}ms | lr {optim.param_groups[0]['lr']:.6f}"
        )

logger.close()
