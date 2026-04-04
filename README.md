# DLOps Assignment 5 — LoRA Fine-tuning & Adversarial Attacks

> **WandB:** https://wandb.ai/YOUR_ENTITY/DLOps-Ass5-Q1  
> **HuggingFace:** https://huggingface.co/YOUR_USERNAME/vit-small-lora-cifar100

---

## Repository Structure

```
Assignment5/
├── Q1/
│   ├── train.py             # ViT-S baseline + LoRA fine-tuning
│   ├── optuna_search.py     # Optuna HPO for LoRA hyperparams
│   └── run_experiments.sh   # One-click script for all Q1 runs
├── Q2/
│   ├── q2i_fgsm.py          # ResNet-18 train + FGSM scratch vs ART
│   └── q2ii_detector.py     # ResNet-34 adversarial detector (PGD, BIM)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Docker Setup (Required)

All experiments must be run inside Docker.

### Build & Run

```bash
# From the root of the repo:
cd docker

# Set your WandB key
export WANDB_API_KEY=your_wandb_api_key_here

# Build the image
docker-compose build

# Launch interactive container
docker-compose run assignment5 bash
```

---

## Q1: ViT-S LoRA Fine-tuning on CIFAR-100

### Install

```bash
pip install -r requirements.txt
```

### Run all experiments

```bash
cd Q1
bash run_experiments.sh
```

### Individual training commands

**Baseline (head-only):**
```bash
python train.py \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir checkpoints
```

**LoRA (example: rank=4, alpha=8):**
```bash
python train.py \
    --use_lora \
    --rank 4 \
    --alpha 8 \
    --dropout 0.1 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir checkpoints
```

**Optuna HPO:**
```bash
python optuna_search.py --n_trials 20
```

**Optional — Partial Freeze + LoRA:**
```bash
python train.py \
    --use_lora \
    --partial_freeze \
    --rank 4 \
    --alpha 8 \
    --dropout 0.1 \
    --epochs 10 \
    --save_dir checkpoints
```

---

## Q1 Results

### Testing Table

| LoRA | Rank | Alpha | Dropout | Test Acc (%) | Trainable Params |
|------|------|-------|---------|--------------|-----------------|
| No   | –    | –     | –       | ~XX.XX       | ~XXX,XXX        |
| Yes  | 2    | 2     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 2    | 4     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 2    | 8     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 4    | 2     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 4    | 4     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 4    | 8     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 8    | 2     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 8    | 4     | 0.1     | ~XX.XX       | ~XX,XXX         |
| Yes  | 8    | 8     | 0.1     | ~XX.XX       | ~XX,XXX         |

> Fill in the actual values after running experiments.

---

## Q2: Adversarial Attacks (IBM ART)

### Q2(i) — FGSM: Scratch vs IBM ART

**Train ResNet-18 and run FGSM:**
```bash
cd Q2
python q2i_fgsm.py \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.1 \
    --save_dir checkpoints_q2i
```

### Q2(ii) — Adversarial Detection

**Train PGD & BIM detectors (ResNet-34):**
```bash
python q2ii_detector.py \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-3 \
    --victim_ckpt checkpoints_q2i/resnet18_best.pt \
    --save_dir checkpoints_q2ii
```

---

## Q2 Results

### FGSM Accuracy Comparison

| Attack       | Epsilon | Accuracy (%) |
|--------------|---------|--------------|
| Clean        | –       | ~XX.XX       |
| FGSM Scratch | 0.03    | ~XX.XX       |
| FGSM ART     | 0.03    | ~XX.XX       |

### Detection Accuracy

| Attack | Detector | Detection Accuracy (%) |
|--------|----------|------------------------|
| PGD    | ResNet-34 | ≥70%                  |
| BIM    | ResNet-34 | ≥70%                  |

---

## Notes

- All experiments use mixed-precision training (AMP).
- CIFAR-100 LoRA experiments use ViT-S (`WinKawaks/vit-small-patch16-224`).
- All adversarial attacks use IBM ART with CIFAR-10 normalization pre-processing.
- Best Q1 model is pushed to HuggingFace after Optuna search.
