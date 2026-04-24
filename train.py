"""
Self-Pruning Neural Network — Tredence AI Engineering Case Study
================================================================
Author  : <Your Name>
Dataset : CIFAR-10
Task    : Image Classification with learnable weight-gate sparsity

Overview
--------
Each weight in every linear layer is paired with a learnable scalar "gate"
(passed through Sigmoid to stay in [0,1]).  An L1 penalty on these gates
pushes them toward 0, effectively pruning the associated weights.

Run
---
    python train.py

Results are printed to stdout and saved to:
    results/gate_distribution_lambda_<value>.png
    results/summary_table.csv
"""

import os
import csv
import math
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear Layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear whose weights are element-wise
    multiplied by learnable gates ∈ (0, 1).

    Forward pass
    ------------
        gates        = sigmoid(gate_scores)          # shape: (out, in)
        pruned_weight = weight * gates               # element-wise
        output        = input @ pruned_weight.T + bias

    The gate_scores tensor is registered as a model parameter so the
    optimiser updates it alongside the weights.  Gradients flow through
    both weight and gate_scores thanks to autograd tracking the
    element-wise product.

    Parameters
    ----------
    in_features  : int  — size of each input sample
    out_features : int  — size of each output sample
    bias         : bool — whether to add a learnable bias (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard weight & bias ─────────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Gate scores (same shape as weight) ────────────────────────────
        # Initialised near 0 so sigmoid(gate_scores) ≈ 0.5 — neither fully
        # open nor closed at the start of training.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialise weight with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def gates(self) -> torch.Tensor:
        """Return the gate values ∈ (0,1) derived from gate_scores."""
        return torch.sigmoid(self.gate_scores)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.gates                  # (out_features, in_features)
        pruned_weight = self.weight * gates         # element-wise product
        return F.linear(x, pruned_weight, self.bias)

    # ── Extra info ────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Self-Pruning Network
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A simple feed-forward network for CIFAR-10 (32×32×3 → 10 classes).

    Architecture
    ------------
        Flatten  →  PrunableLinear(3072, 1024)  →  BN  →  ReLU  →  Dropout
                 →  PrunableLinear(1024, 512)   →  BN  →  ReLU  →  Dropout
                 →  PrunableLinear(512, 256)    →  BN  →  ReLU  →  Dropout
                 →  PrunableLinear(256, 10)

    All linear layers use PrunableLinear so every weight has its own gate.
    Batch-norm and dropout provide additional regularisation independent of
    the sparsity mechanism.
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.ModuleList([
            PrunableLinear(3 * 32 * 32, 1024),
            PrunableLinear(1024, 512),
            PrunableLinear(512,  256),
            PrunableLinear(256,  10),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
        ])

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for i, layer in enumerate(self.layers[:-1]):      # hidden layers
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)                            # output layer
        return x

    # ── Gate helpers ──────────────────────────────────────────────────────

    def all_gates(self) -> List[torch.Tensor]:
        """Return a list of gate tensors from every PrunableLinear layer."""
        return [layer.gates for layer in self.layers]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values — encourages gates to collapse to 0.
        Because gates = sigmoid(scores) ∈ (0,1) the absolute-value in the
        L1 norm is redundant, so this is simply the sum of all gate values.
        """
        return sum(g.sum() for g in self.all_gates())

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate is below `threshold` (treated as pruned).
        Returns a value in [0, 1].
        """
        all_gate_vals = torch.cat([g.detach().flatten() for g in self.all_gates()])
        pruned        = (all_gate_vals < threshold).float().mean().item()
        return pruned


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    data_dir : str  = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return train / test DataLoaders for CIFAR-10.
    Applies standard normalisation with CIFAR-10 channel statistics.
    """
    # CIFAR-10 per-channel mean and std (computed on training set)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model      : SelfPruningNet,
    loader     : DataLoader,
    optimizer  : optim.Optimizer,
    device     : torch.device,
    lambda_    : float,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Total Loss = CrossEntropyLoss(logits, labels) + λ × SparsityLoss

    Returns
    -------
    avg_cls_loss     : mean cross-entropy over all batches
    avg_total_loss   : mean total loss over all batches
    """
    model.train()
    total_cls  = 0.0
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits         = model(images)
        cls_loss       = F.cross_entropy(logits, labels)
        sparse_loss    = model.sparsity_loss()
        loss           = cls_loss + lambda_ * sparse_loss

        loss.backward()
        optimizer.step()

        total_cls  += cls_loss.item()
        total_loss += loss.item()

    n = len(loader)
    return total_cls / n, total_loss / n


@torch.no_grad()
def evaluate(
    model  : SelfPruningNet,
    loader : DataLoader,
    device : torch.device,
) -> float:
    """Return top-1 accuracy on the provided DataLoader."""
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(
    model     : SelfPruningNet,
    lambda_   : float,
    save_dir  : str = "results",
) -> None:
    """
    Save a histogram of all gate values for the trained model.

    A successful pruning run shows a large spike near 0 (pruned weights)
    and a secondary cluster away from 0 (surviving weights).
    """
    os.makedirs(save_dir, exist_ok=True)

    all_gates = (
        torch.cat([g.detach().cpu().flatten() for g in model.all_gates()])
        .numpy()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gates, bins=100, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count",      fontsize=12)
    ax.set_title(
        f"Gate Value Distribution  (λ = {lambda_})\n"
        f"Sparsity = {model.sparsity_level()*100:.1f}%",
        fontsize=13,
    )
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=10)
    plt.tight_layout()

    fname = os.path.join(save_dir, f"gate_distribution_lambda_{lambda_}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [plot] saved → {fname}")


def plot_accuracy_sparsity_tradeoff(
    results   : List[dict],
    save_dir  : str = "results",
) -> None:
    """
    Plot accuracy vs sparsity across different λ values on a dual-axis chart.
    """
    os.makedirs(save_dir, exist_ok=True)

    lambdas   = [r["lambda"]   for r in results]
    accs      = [r["accuracy"] * 100 for r in results]
    sparsities= [r["sparsity"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "steelblue"
    color2 = "darkorange"

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color=color1, fontsize=12)
    ax1.plot(lambdas, accs, "o-", color=color1, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity Level (%)", color=color2, fontsize=12)
    ax2.plot(lambdas, sparsities, "s--", color=color2, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Accuracy vs Sparsity  (λ Trade-off)", fontsize=13)
    fig.tight_layout()

    fname = os.path.join(save_dir, "accuracy_vs_sparsity.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [plot] saved → {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main Experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lambda_    : float,
    epochs     : int,
    lr         : float,
    device     : torch.device,
    train_loader: DataLoader,
    test_loader : DataLoader,
    save_dir   : str = "results",
) -> dict:
    """
    Train a fresh SelfPruningNet for a given λ, evaluate, and return a
    result dict.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lambda_}   |   epochs = {epochs}   |   lr = {lr}")
    print(f"{'='*60}")

    model     = SelfPruningNet(dropout_p=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        cls_loss, total_loss = train_one_epoch(
            model, train_loader, optimizer, device, lambda_
        )
        scheduler.step()

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, device)
            sp  = model.sparsity_level()
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  "
                f"cls_loss={cls_loss:.4f}  total_loss={total_loss:.4f}  |  "
                f"val_acc={acc*100:.2f}%  sparsity={sp*100:.1f}%"
            )

    elapsed = time.time() - t0
    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()

    print(f"\n  ✓ Final  Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓ Final  Sparsity : {final_sparsity*100:.1f}%")
    print(f"  ✓ Time elapsed    : {elapsed:.1f}s")

    plot_gate_distribution(model, lambda_, save_dir)

    return {
        "lambda"  : lambda_,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
    }


def save_csv(results: List[dict], save_dir: str = "results") -> None:
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, "summary_table.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "accuracy", "sparsity"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "lambda"  : r["lambda"],
                "accuracy": f"{r['accuracy']*100:.2f}%",
                "sparsity": f"{r['sparsity']*100:.1f}%",
            })
    print(f"\n  [csv] summary saved → {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — CIFAR-10"
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[1e-5, 1e-4, 5e-4],
        help="List of λ values to sweep (default: 1e-5 1e-4 5e-4)"
    )
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Training epochs per λ (default: 30)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int,   default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--data-dir",   type=str,   default="./data",
                        help="Directory to download CIFAR-10 (default: ./data)")
    parser.add_argument("--save-dir",   type=str,   default="./results",
                        help="Directory for plots & CSV (default: ./results)")
    parser.add_argument("--num-workers",type=int,   default=2)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] Using: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    all_results = []
    for lam in args.lambdas:
        result = run_experiment(
            lambda_      = lam,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            train_loader = train_loader,
            test_loader  = test_loader,
            save_dir     = args.save_dir,
        )
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lambda':<12}  {'Test Accuracy':>14}  {'Sparsity (%)':>13}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*13}")
    for r in all_results:
        print(
            f"  {r['lambda']:<12}  "
            f"{r['accuracy']*100:>13.2f}%  "
            f"{r['sparsity']*100:>12.1f}%"
        )
    print(f"{'='*60}\n")

    save_csv(all_results, args.save_dir)
    plot_accuracy_sparsity_tradeoff(all_results, args.save_dir)

    print("\n[done] All experiments complete. Check the `results/` folder.\n")


if __name__ == "__main__":
    main()
