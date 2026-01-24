# Assignment 1 – ML-DL-Ops

## Student Details
- **Name:** Sushantak Parashar Jha  
- **Roll No:** M25CSA035  
- **Program:** M.Tech in Artificial Intelligence  
- **Department:** Computer Science and Engineering  
- **Institution:** Indian Institute of Technology Jodhpur  
- **Email:** m25csa035@iitj.ac.in  

---

## Colab Notebook
The complete Colab notebook containing **already run experiments and outputs** is available here:

👉 [Colab Notebook Link](https://drive.google.com/file/d/1o43FA3qdoDlyZaTmlM4lJMYXPI9r8B-n/view?usp=drive_link)

---

## Q1(a) Deep Learning Experiments

### MNIST Results

Deep learning experiments were performed using **ResNet-18** and **ResNet-50** with different optimizers, learning rates, batch sizes, and epochs.

#### Key Observations
- Adam optimizer consistently outperformed SGD.
- Learning rate **0.001** performed significantly better than **0.0001**.
- Increasing epochs from 5 to 10 yielded marginal gains.
- Larger batch sizes slightly reduced accuracy for SGD but had minimal impact with Adam.

#### MNIST Results Table

| Batch Size | Optimizer | Learning Rate | Epochs | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|-----------|----------|---------------|--------|------------------|------------------|
| 16 | SGD | 0.001 | 5 | 97.90 | 97.05 |
| 16 | SGD | 0.0001 | 5 | 74.67 | 51.21 |
| 16 | Adam | 0.001 | 5 | 99.00 | 98.80 |
| 16 | Adam | 0.001 | 10 | 99.27 | — |
| 16 | Adam | 0.0001 | 5 | 99.07 | 99.07 |
| 32 | SGD | 0.001 | 5 | 96.15 | 94.28 |
| 32 | Adam | 0.0001 | 5 | 99.27 | 99.11 |

**Best MNIST Model:**  
ResNet-18 with Adam, learning rate 0.001, batch size 32 (99.27%).

---

### FashionMNIST Results

FashionMNIST experiments were more challenging due to higher intra-class similarity.

#### Key Observations
- ResNet-50 outperformed ResNet-18 when trained with Adam.
- SGD with low learning rate converged poorly.
- Adam showed strong robustness across architectures.

#### FashionMNIST Results Table

| Batch Size | Optimizer | Learning Rate | Epochs | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|-----------|----------|---------------|--------|------------------|------------------|
| 16 | SGD | 0.001 | 5 | 78.67 | 81.62 |
| 16 | SGD | 0.0001 | 5 | 72.75 | 48.23 |
| 16 | Adam | 0.001 | 5 | 92.99 | 98.78 |
| 16 | Adam | 0.0001 | 5 | 92.01 | 99.18 |
| 32 | SGD | 0.001 | 5 | 98.16 | 97.37 |
| 32 | Adam | 0.0001 | 5 | 99.19 | — |

**Best FashionMNIST Model:**  
ResNet-50 with Adam, learning rate 0.0001 (99.18%).

---

## Q1(b) SVM Experiments

Support Vector Machines were evaluated on reduced subsets due to high computational cost.

### Observations
- Polynomial kernel slightly outperformed RBF on both datasets.
- Performance decreased on FashionMNIST due to increased complexity.
- CNN models consistently outperformed SVMs.

#### SVM Results Table

| Dataset | Kernel | Train Samples | Test Samples | Accuracy (%) |
|-------|--------|---------------|--------------|--------------|
| MNIST | RBF | 3000 | 1000 | 93.90 |
| MNIST | Polynomial | 3000 | 1000 | 94.20 |
| FashionMNIST | RBF | 3000 | 1000 | 79.60 |
| FashionMNIST | Polynomial | 3000 | 1000 | 81.10 |

---

## Q2 Hardware Performance Analysis (CPU vs GPU)

### Key Observations
- GPU significantly reduces training time, enabling multi-epoch training.
- Classification accuracy remains identical across hardware.
- FLOPs depend solely on model architecture.
- SVM experiments were restricted to CPU due to library limitations.

### ResNet-18 Hardware Comparison

| Compute | Batch | Epochs | Optimizer | Accuracy (%) | Train Time (ms) | FLOPs |
|------|------|--------|----------|--------------|----------------|-------|
| CPU | 16 | 3 | SGD | 87.62 | 4,100,938.93 | 1.82 G |
| CPU | 16 | 3 | Adam | 88.63 | 5,421,659.86 | 1.82 G |
| GPU | 16 | 5 | SGD | 97.90 | 545,332.58 | 1.74 G |
| GPU | 16 | 5 | Adam | 99.00 | 545,332.58 | 1.74 G |

### ResNet-50 Hardware Comparison

| Compute | Batch | Epochs | Optimizer | Accuracy (%) | Train Time (ms) |
|------|------|--------|----------|--------------|----------------|
| CPU | 16 | 1 | SGD | — | — |
| CPU | 16 | 1 | Adam | — | — |
| GPU | 16 | 5 | SGD | 97.05 | 1,074,583.06 |
| GPU | 16 | 5 | Adam | 98.80 | 1,160,030.18 |

---

## Overall Conclusion
- Deep learning models significantly outperform classical SVMs.
- Adam with learning rate 0.001 provides the best convergence.
- ResNet-18 balances accuracy and efficiency, while ResNet-50 achieves peak performance.
- GPU acceleration is essential for practical deep learning workloads.

