# Self-Pruning Neural Network for CIFAR-10

This project implements a **self-pruning convolutional neural network** that learns to remove unnecessary connections during training using **differentiable gating and sparsity regularization**.

Unlike traditional pruning methods applied after training, this approach enables the model to **adapt its architecture dynamically**, improving efficiency while maintaining competitive accuracy.

---

## Overview

Each weight in the prunable layers is paired with a learnable gate:

```
w_eff = w × sigmoid(g)
```

* `sigmoid(g)` acts as a soft gate in the range (0, 1)
* When the gate approaches 0, the corresponding connection is effectively pruned
* Gates are optimized jointly with weights using gradient descent

---

## Architecture

```
Input (3×32×32)
   │
   ├── CNN Backbone (Conv → BatchNorm → ReLU)
   │       ├── Stage 1 → MaxPool
   │       ├── Stage 2 → MaxPool
   │       └── Stage 3 → AvgPool
   │
   └── Prunable Classifier
           ├── PrunableLinear(2048 → 512)
           ├── PrunableLinear(512 → 256)
           └── PrunableLinear(256 → 10)
```

Pruning is applied only to the fully connected layers, where parameter density is highest and pruning is most effective.

---

## Loss Function

```
L = CE + λ × mean(sigmoid(g))
```

* `CE` is the cross-entropy loss
* `λ` controls sparsity strength
* `mean(sigmoid(g))` encourages gates to move toward zero

### Key Design Choice: Normalized Sparsity Loss

The sparsity term is computed as the **mean** of gate activations instead of the sum.

This ensures:

* Comparable scale between classification and sparsity losses
* Stable gradients
* Effective control of pruning via λ

---

## Training Pipeline

### Phase 1: Soft Training

* Gates remain continuous in (0, 1)
* λ is gradually increased (curriculum sparsity)
* Model learns useful features before pruning begins

### Phase 2: Hard Thresholding

* Gates below threshold τ are set to zero
* Produces an explicitly sparse model

### Phase 3: Fine-Tuning

* Retraining with λ = 0
* Recovers accuracy lost due to pruning

---

## Experimental Results

| λ    | Test Accuracy | Sparsity | MAC Reduction |
| ---- | ------------- | -------- | ------------- |
| 0.01 | 85.4%         | 8.1%     | 8.1%          |
| 0.10 | 80.9%         | 32.1%    | 32.1%         |
| 0.40 | 72.8%         | 67.9%    | 67.9%         |

### Observations

* Increasing λ increases sparsity while reducing accuracy
* Gate distributions become bimodal, indicating clear pruning decisions
* Fine-tuning recovers performance after hard pruning
* A balanced trade-off is achieved at moderate λ values

---

## Key Contributions

* Differentiable pruning using learnable gates
* Normalized sparsity loss to resolve loss-scale imbalance
* Curriculum-based λ scheduling for stable training
* Three-phase pruning pipeline (soft → hard → fine-tune)
* Empirical validation with accuracy-sparsity trade-off

---

## Project Structure

```
├── model.py        # PrunableLinear and CNN architecture
├── train.py        # Training, pruning, and fine-tuning pipeline
├── utils.py        # Data loading, loss functions, metrics
├── visualize.py    # Plots and analysis
├── main.py         # Entry point
└── report.md       # Detailed technical report
```

---

## Setup

### Install dependencies

```
pip install torch torchvision matplotlib numpy
```

### Run training

```
python main.py
```

---

## Device Support

* CPU
* CUDA (NVIDIA GPU)
* Apple Silicon (MPS, with minor adjustments)

---

## Limitations

* Pruning is limited to fully connected layers
* Unstructured sparsity may not fully translate to hardware speedups
* Experiments are conducted on CIFAR-10 only

---

## Future Work

* Structured pruning (channel/filter-level)
* Sparse inference optimization
* Extension to larger datasets (e.g., ImageNet)
* Exploration of binary or stochastic gating

---

## Author

Ranesh Prashar

---

## License

This project is intended for academic and educational use.
