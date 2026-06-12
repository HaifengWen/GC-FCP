# medmnist_train3d.py
# Train a simple 3D CNN on OrganMNIST3D (28x28x28, 11 classes) and save a checkpoint.
# Usage (example):
#   python medmnist_train3d.py --epochs 15 --batch-size 128 --lr 1e-3 --out checkpoints/organ3d_cnn.pt
#
# Requirements:
#   pip install torch torchvision medmnist tqdm
#
# Notes:
# - We keep the architecture minimal but stable for 28^3 inputs.
# - The saved checkpoint contains: {'model_state': state_dict, 'meta': {...}} so the experiment script can load it.

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import medmnist
import torchvision.transforms as T
from medmnist import INFO

DATA_FLAG = 'pathmnist'
N_CLASSES = 9  # per MedMNIST

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_datasets(download: bool = True):
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    # medmnist returns np arrays in [0, 255] for 2D. We convert to float32 and add channel.
    train_ds = DataClass(split='train', download=download, root='data', transform=T.Compose([T.ToTensor()]), target_transform=lambda y: int(y))
    val_ds   = DataClass(split='val', download=download, root='data', transform=T.Compose([T.ToTensor()]), target_transform=lambda y: int(y))
    test_ds  = DataClass(split='test', download=download, root='data', transform=T.Compose([T.ToTensor()]), target_transform=lambda y: int(y))
    return train_ds, val_ds, test_ds

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

def run(args):
    os.makedirs(Path(args.out).parent, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    train_ds, val_ds, _ = get_datasets(download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SimpleCNN(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = torch.tensor(y, dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        nval = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.tensor(y, dtype=torch.long, device=device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_acc  += (logits.argmax(1) == y).float().sum().item()
                nval     += x.size(0)
        val_loss /= nval
        val_acc /= nval

        print(f"[val] loss={val_loss:.4f}, acc={val_acc:.4f} | [train] loss={train_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'meta': {
                    'n_classes': N_CLASSES,
                    'arch': 'SimpleCNN',
                    'val_acc': best_acc,
                }
            }, args.out)
            print(f"  -> saved new best to {args.out} (val_acc={best_acc:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--out', type=str, default='checkpoints/path_cnn.pt')
    args = parser.parse_args()
    run(args)
