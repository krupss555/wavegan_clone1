"""
eval/train_classifier.py
────────────────────────
Train a lightweight 1D-CNN on REAL DeepRadar signals.
This classifier is later used by inception_score.py and similarity.py
to extract intermediate features for FID / KNN evaluation.

Architecture: 3× Conv1d → GAP → FC → 23-class softmax
Mirrors the Discriminator backbone but with classification head.

Run
───
python eval/train_classifier.py \
    --data_dir  /path/to/data/DeepRadar \
    --out       ./runs/eval/classifier.pt \
    --epochs    50 \
    --batch     128 \
    --lr        1e-3 \
    --num_workers 4
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reach parent dir for dataset import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DeepRadarDataset


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class RadarClassifier(nn.Module):
    """
    Input : (B, 2, 128)
    Output: (B, num_classes)  raw logits

    Intermediate feature layer (used for FID / KNN) :
      call .features(x) → (B, 256) after global-average-pool
    """
    def __init__(self, num_classes=23):
        super().__init__()
        # same kernel/stride as discriminator so features are comparable
        self.conv0 = nn.Conv1d(  2,  64, kernel_size=25, stride=4, padding=12)
        self.bn0   = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d( 64, 128, kernel_size=25, stride=4, padding=12)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=12)
        self.bn2   = nn.BatchNorm1d(256)
        # global-average-pool → 256-dim feature
        self.fc    = nn.Linear(256, num_classes)

    def features(self, x):
        """Returns (B, 256) intermediate feature vector."""
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2)   # (B, 64,  32)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)   # (B, 128,  8)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)   # (B, 256,  2)
        x = x.mean(dim=-1)                                 # (B, 256) GAP
        return x

    def forward(self, x):
        return self.fc(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classifier] device={device}")

    train_ds  = DeepRadarDataset(args.data_dir, split="train")
    val_ds    = DeepRadarDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = RadarClassifier(num_classes=23).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for epoch in range(args.epochs):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                preds  = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"loss {avg_loss:.4f} | val_acc {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(),
                        "num_classes": 23,
                        "val_acc": best_acc}, args.out)
            print(f"  ↳ Saved best classifier → {args.out}  (val_acc={best_acc*100:.2f}%)")

    print(f"\n[Classifier] Best val accuracy: {best_acc*100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--out",         default="./runs/eval/classifier.pt")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch",       type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=4)
    args = p.parse_args()
    train(args)