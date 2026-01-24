# Assignment 1 – ML-DL-Ops (CS6001)

## Student Information
- **Name:** Sushantak Parashar Jha  
- **Roll No:** M25CSA035  
- **Program:** M.Tech in AI
- **Department:** Computer Science and Engineering  

---

## Colab Notebook (Already Run)
The complete Colab notebook containing all experiments **with outputs and plots already executed** is available here:

👉 **[Colab Notebook Link](https://colab.research.google.com/drive/1-2cC7Kq-krFh-D41jr8g4EC7izknplRE#scrollTo=X4B3P9JFRGA5)**

---

## 1. Deep Learning Experiments (Q1a)

### 1.1 Methodology and Architectures
The objective of this task was to evaluate the robustness of **Residual Networks (ResNet)** on two benchmark datasets:
- **MNIST** (handwritten digits)
- **FashionMNIST** (clothing articles)

Two architectures were used:
- **ResNet-18:** A lightweight architecture (~11M parameters) suitable for faster experimentation.
- **ResNet-50:** A deeper architecture with bottleneck blocks (~23M parameters), designed to learn complex hierarchical features.

---

### 1.2 Results on MNIST

Experiments were conducted by varying **optimizers (SGD, Adam)**, **learning rates**, **batch sizes**, and **epochs**.

#### Technical Observations
- Adam consistently outperformed SGD, achieving faster convergence.
- Learning rate **0.001** performed significantly better than **0.0001**.
- Increasing epochs from 5 to 10 yielded only marginal gains.
- Larger batch size slightly degraded SGD performance but had minimal effect with Adam.

#### Accuracy Below 80% and Epoch Consideration
Some configurations using **SGD with a low learning rate (0.0001)** resulted in accuracy below 80%. Although increasing epochs improved validation accuracy, even at 10 epochs the performance remained below 80%, indicating suboptimal hyperparameter selection. These configurations were therefore not extended further due to limited additional insight and high computational cost.

#### Table 1: MNIST Results

| Batch Size | Optimizer | LR | Epochs | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|----|--------|---------------|---------------|
| 16 | SGD | 0.001 | 5 | 97.90 | 97.05 |
| 16 | SGD | 0.0001 | 5 / 10 | 74.67 | 72.47 |
| 16 | Adam | 0.001 | 5 | 99.00 | 98.80 |
| 16 | Adam | 0.001 | 10 | 99.27 | — |
| 16 | Adam | 0.0001 | 5 | 99.07 | 99.07 |
| 32 | SGD | 0.001 | 5 | 96.15 | 94.28 |
| 32 | Adam | 0.0001 | 5 | 99.27 | 99.11 |

**Best MNIST Configuration:**  
ResNet-18 + Adam, LR = 0.001, Batch Size = 32 (99.27%)

---

### 1.3 Results on FashionMNIST

FashionMNIST is more challenging due to higher intra-class similarity and visual complexity.

#### Technical Observations
- ResNet-50 outperformed ResNet-18 when trained with Adam.
- SGD with LR = 0.0001 converged poorly, especially for deeper models.
- Adam showed strong robustness across architectures.

#### Table 2: FashionMNIST Results

| Batch Size | Optimizer | LR | Epochs | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|----|--------|---------------|---------------|
| 16 | SGD | 0.001 | 5 | 78.67 | 81.62 |
| 16 | SGD | 0.0001 | 5 / 10 | 72.75 | 70.84% |
| 16 | Adam | 0.001 | 5 | 92.99 | 98.78 |
| 16 | Adam | 0.0001 | 5 | 92.01 | 99.18 |
| 32 | SGD | 0.001 | 5 | 98.16 | 97.37 |
| 32 | Adam | 0.0001 | 5 | 99.19 | 98.85 |

**Best FashionMNIST Configuration:**  
ResNet-50 + Adam, LR = 0.0001 (99.18%)

---

## 2. SVM Performance Analysis (Q1b)

Support Vector Machines were evaluated using **RBF** and **Polynomial kernels** on reduced subsets due to computational cost.

#### Observations
- Polynomial kernel slightly outperformed RBF on both datasets.
- Performance dropped on FashionMNIST due to increased complexity.
- CNN models significantly outperformed SVMs.

#### Table 3: SVM Results

| Dataset | Kernel | Train Samples | Test Samples | Accuracy (%) |
|--------|--------|---------------|--------------|--------------|
| MNIST | RBF | 3000 | 1000 | 93.90 |
| MNIST | Polynomial | 3000 | 1000 | 94.20 |
| FashionMNIST | RBF | 3000 | 1000 | 79.60 |
| FashionMNIST | Polynomial | 3000 | 1000 | 81.10 |

---

## 3. Hardware Performance Analysis (Q2)

This section evaluates the **Ops** aspect of ML-DL-Ops by comparing CPU and GPU execution.

### Key Findings
- GPU offered ~7.5× speedup for ResNet-18 training.
- FLOPs remain constant across hardware (architecture dependent).
- CPU training becomes impractical for deeper models.
- SVMs were limited to CPU due to library constraints.

### ResNet-18 Hardware Comparison

| Compute | Epochs | Optimizer | Accuracy (%) | Train Time (ms) | FLOPs |
|--------|--------|----------|--------------|-----------------|-------|
| CPU | 3 | SGD | 87.62 | 4,100,938.93 | 1.82 G |
| CPU | 3 | Adam | 88.63 | 5,421,659.86 | 1.82 G |
| GPU | 5 | SGD | 97.90 | 545,332.58 | 1.74 G |
| GPU | 5 | Adam | 99.00 | 545,332.58 | 1.74 G |

### ResNet-50 Hardware Comparison

| Compute | Epochs | Optimizer | Accuracy (%) | Train Time (ms) |
|--------|--------|----------|--------------|-----------------|
| CPU | 1 | SGD | 76.93 | 2,495,743.50 |
| CPU | 1 | Adam | 83.34 | 3,210,639.37 |
| GPU | 5 | SGD | 97.05 | 1,074,583.06 |
| GPU | 5 | Adam | 98.80 | 1,160,030.18 |

---

## 4. Conclusion

- Adaptive optimizers like **Adam** are crucial for efficient deep learning.
- **ResNet-18** provides a strong balance between accuracy and efficiency.
- **ResNet-50** achieves superior accuracy for complex datasets.
- **GPU acceleration** is essential for modern deep learning workflows.

---

## GitHub Pages
This repository is published using **GitHub Pages** and is updated after each assignment submission.
