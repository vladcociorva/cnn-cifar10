import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from model import CNN

def tta(image: torch.Tensor) -> torch.Tensor:
    original = image
    flipped = torch.flip(image, dims=[2])
    return torch.stack([original, flipped])

parser = argparse.ArgumentParser(description="Test CNN model on CIFAR-10")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--use-tta", action="store_true", help="Use test-time augmentation (horizontal flip)")
args = parser.parse_args()

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)
device = torch.device("cuda:0")

model = CNN()
model.to(device)
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Using {'test-time augmentation' if args.use_tta else 'standard testing'}")

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])
)
batch_size = 1 if args.use_tta else 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        if args.use_tta:
            augmented = tta(x.squeeze(0))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(augmented)
                avg_logits = torch.mean(logits, dim=0, keepdim=True)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                avg_logits, _ = model(x)
        
        pred = torch.argmax(avg_logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test accuracy: {correct/total:.4f}")
