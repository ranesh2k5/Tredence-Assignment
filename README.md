# Self-Pruning Neural Network with Learnable Gates

## Overview

This project implements a self-pruning neural network for image classification on CIFAR-10. The model learns both its weights and its structure simultaneously by associating each connection with a learnable gate. During training, these gates are optimized to suppress unimportant connections, resulting in a sparse and efficient network.

Unlike traditional pruning methods applied after training, this approach integrates pruning directly into the optimization process.

---

## Key Idea

Each weight in the fully connected layers is paired with a learnable gate:

w_eff = w × sigmoid(g)

* The sigmoid function maps gate scores to (0, 1)
* Values close to 0 effectively remove the connection
* Values close to 1 retain the connection

The model is trained with a combined objective:

L = CrossEntropyLoss + λ × SparsityLoss

Where:

* CrossEntropyLoss ensures classification accuracy
* SparsityLoss encourages gates to move toward zero

---

## Sparsity Mechanism

A log-based sparsity penalty is used:

SparsityLoss = mean( -log(1 - sigmoid(g) + ε) )

This formulation:

* Applies stronger gradients to large gate values
* Avoids vanishing gradients seen in standard L1 penalties
* Encourages a bimodal distribution (near 0 or 1)

---

## Model Architecture

The model consists of:

* A CNN feature extractor (Conv + BatchNorm + ReLU)
* A prunable fully connected classifier

Structure:

* Convolutional backbone (3 stages)
* Flatten layer
* PrunableLinear (2048 → 512)
* PrunableLinear (512 → 256)
* PrunableLinear (256 → 10)

Only the fully connected layers are pruned, as they contain the majority of parameters.

Implementation details: 

---

## Training Pipeline

The training process is divided into three phases:

### 1. Soft-Gate Training

* Gates are continuous (0–1)
* Sparsity penalty is gradually introduced using a λ schedule
* The model learns both features and connection importance

### 2. Hard Pruning

* Gates below a threshold τ are set to zero
* Corresponding weights are permanently disabled

### 3. Fine-Tuning

* Training continues with λ = 0
* Helps recover any accuracy lost during pruning

Implementation details: 

---

## Lambda Scheduling

λ is gradually increased during early training:

* Prevents premature pruning
* Allows the model to first learn useful representations

Typical schedule:

* Warm-up phase: λ increases linearly
* Later epochs: λ remains constant

Implementation: 

---

## Dataset

* CIFAR-10 (50,000 training, 10,000 test samples)
* Data augmentation:

  * Random crop
  * Horizontal flip
  * Color jitter

Loader implementation: 

---

## Experiments

Three levels of sparsity were evaluated:

| λ Value | Behavior           |
| ------- | ------------------ |
| 0.01    | Light pruning      |
| 0.10    | Moderate pruning   |
| 0.40    | Aggressive pruning |

Each experiment follows the full pipeline:
training → pruning → fine-tuning

Entry point: 

---

## Metrics Reported

* Test Accuracy
* Sparsity (% of pruned connections)
* MAC Reduction (theoretical compute savings)
* Accuracy before and after pruning

---

## Results Summary

* Increasing λ leads to higher sparsity
* Higher sparsity reduces computational cost
* Moderate λ provides the best trade-off between accuracy and efficiency
* Fine-tuning consistently recovers performance after pruning

---

## Visualization

The project generates a comprehensive visualization including:

* Training and test accuracy curves
* Gate value distributions
* Pruning impact on accuracy
* Accuracy vs sparsity trade-off
* MAC reduction comparison

Visualization code: 

---

## How to Run

```bash
python main.py
```

Outputs:

* Console logs with training progress
* Summary table across experiments
* Final visualization saved as `pruning_results.png`

---

## Project Structure

```
main.py        → experiment orchestration
model.py       → prunable layers and CNN model
train.py       → training, pruning, fine-tuning pipeline
utils.py       → loss functions, data loading, metrics
visualize.py   → result plots
```

---

## Key Contributions

* End-to-end self-pruning neural network
* Differentiable gating mechanism
* Log-based sparsity loss for effective pruning
* Curriculum-based sparsity scheduling
* Complete training → pruning → fine-tuning pipeline

---

## Conclusion

This project demonstrates that neural networks can learn to optimize their own structure during training. By integrating pruning into the learning process, the model achieves a balance between performance and efficiency without requiring a separate pruning stage.

---

## License

This project is intended for academic and educational use.
