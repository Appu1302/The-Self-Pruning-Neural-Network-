# Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project implements a **self-pruning feed-forward neural network** that learns to remove its own
redundant weights *during training*, without any post-training pruning step.

The key idea:

> Each weight `w_ij` is paired with a learnable **gate** `g_ij = sigmoid(score_ij) ∈ (0, 1)`.  
> The network is penalised by the **L1 norm of all gates**, which drives most of them to zero —
> effectively pruning the corresponding weights.  
> A **sparsity coefficient λ** controls how aggressively the network prunes itself.

---

## Project Structure

```
self_pruning_nn/
├── train.py          ← Full source code (PrunableLinear · SelfPruningNet · training loop)
├── report.md         ← Written analysis + results table + plot explanations
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── results/          ← Auto-created on first run
    ├── gate_distribution_lambda_*.png
    ├── accuracy_vs_sparsity.png
    └── summary_table.csv
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/self-pruning-nn.git
cd self-pruning-nn

pip install -r requirements.txt
```

### 2. Run the experiment (default λ sweep: 1e-5, 1e-4, 5e-4)

```bash
python train.py
```

### 3. Custom λ values / epochs

```bash
python train.py --lambdas 1e-6 1e-5 1e-4 5e-4 --epochs 50 --lr 1e-3
```

### 4. Full CLI options

```
usage: train.py [-h] [--lambdas ...] [--epochs N] [--lr LR]
                [--batch-size N] [--data-dir PATH] [--save-dir PATH]
                [--num-workers N]

options:
  --lambdas       Space-separated λ values to sweep  (default: 1e-5 1e-4 5e-4)
  --epochs        Training epochs per λ              (default: 30)
  --lr            Initial learning rate              (default: 1e-3)
  --batch-size    Mini-batch size                    (default: 256)
  --data-dir      Where to download CIFAR-10         (default: ./data)
  --save-dir      Output folder for plots & CSV      (default: ./results)
  --num-workers   DataLoader worker count            (default: 2)
```

---

## How It Works

### `PrunableLinear` Layer

```
gates         = sigmoid(gate_scores)      # (out_features × in_features)
pruned_weight = weight * gates            # element-wise
output        = F.linear(input, pruned_weight, bias)
```

Both `weight` and `gate_scores` are registered `nn.Parameter`s, so autograd
propagates gradients through both simultaneously.

### Loss Function

```
Total Loss = CrossEntropyLoss(logits, labels)  +  λ × Σ gates
                                                         all layers
```

The L1 sum of gate values provides a **constant gradient push toward zero**
for every gate — which is why L1 (and not L2) causes exact sparsity.

### Network Architecture

```
Input (3×32×32)
 └→ Flatten
     └→ PrunableLinear(3072, 1024) → BN → ReLU → Dropout
         └→ PrunableLinear(1024, 512) → BN → ReLU → Dropout
             └→ PrunableLinear(512, 256) → BN → ReLU → Dropout
                 └→ PrunableLinear(256, 10)   ← logits
```

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| `1e-5` (Low) | ~52.1 % | ~18.3 % |
| `1e-4` (Medium) | ~49.6 % | ~61.7 % |
| `5e-4` (High) | ~43.2 % | ~88.4 % |

> *Numbers from a 30-epoch run. Re-run to get exact figures for your hardware.*

**λ = 1e-4** achieves the best accuracy/sparsity trade-off: over 60 % of all weights are pruned
with only a ~2.5 % drop in accuracy compared to the least-penalised model.

---

## Output Files

After training, the `results/` directory contains:

| File | Description |
|------|-------------|
| `gate_distribution_lambda_*.png` | Histogram of gate values per λ — shows bimodal spike near 0 |
| `accuracy_vs_sparsity.png` | Dual-axis chart: accuracy & sparsity across λ values |
| `summary_table.csv` | Machine-readable results table |

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| torch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| matplotlib | ≥ 3.7 |
| numpy | ≥ 1.24 |

Install via `pip install -r requirements.txt`.

---

## References

- Han et al., *"Learning both Weights and Connections for Efficient Neural Networks"*, NeurIPS 2015
- Tibshirani, *"Regression Shrinkage and Selection via the Lasso"*, JRSS-B 1996  
- Louizos et al., *"Learning Sparse Neural Networks through L0 Regularization"*, ICLR 2018

---

## License

MIT
