# DLOps Assignment 5 — LoRA Fine-tuning & Adversarial Attacks (IBM ART)

**Name:** Sushantak Parashar Jha | **Roll No:** M25CSA035

> **WandB Q1:** https://wandb.ai/sushantak17-prom-iit-rajasthan/DLOps-Ass5-Q1
> **WandB Q1 Optuna:** https://wandb.ai/sushantak17-prom-iit-rajasthan/DLOps-Ass5-Q1-Optuna
> **WandB Q2(i):** https://wandb.ai/sushantak17-prom-iit-rajasthan/DLOps-Ass5-Q2i
> **WandB Q2(ii):** https://wandb.ai/sushantak17-prom-iit-rajasthan/DLOps-Ass5-Q2ii
> **HuggingFace:** https://huggingface.co/Sushantak17/vit-small-lora-cifar100

---

## Repository Structure

```
Assignment5/
├── Q1/
│   ├── train.py               # ViT-S baseline + LoRA fine-tuning
│   ├── optuna_search.py       # Optuna HPO for LoRA hyperparams
│   ├── run_experiments.sh     # Run all Q1 experiments
│   └── push_to_hub.py         # Push best model to HuggingFace
├── Q2/
│   ├── q2i_fgsm.py            # ResNet-18 train + FGSM scratch vs ART
│   └── q2ii_detector.py       # ResNet-34 adversarial detector (PGD, BIM)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── device_utils.py
├── requirements.txt
└── README.md
```

---

## Docker Setup (Required)

All experiments must be run inside Docker.

### Build & Run

```bash
# From the root of the repo
docker build -f docker/Dockerfile -t dlops-ass5 .

# Set your WandB key and run container
docker run -it \
  --name dlops_ass5 \
  -v /path/to/DLOps-Assignment5:/workspace \
  -e WANDB_API_KEY=your_wandb_api_key_here \
  dlops-ass5 bash
```

---

## Q1: ViT-S LoRA Fine-tuning on CIFAR-100

### Install

```bash
pip install -r requirements.txt
```

### Training Commands

**Baseline (head-only fine-tuning):**
```bash
cd Q1
python train.py --epochs 10 --batch_size 64 --lr 1e-3 --save_dir checkpoints
```

**LoRA (rank=2, alpha=2):**
```bash
python train.py --use_lora --rank 2 --alpha 2 --dropout 0.1 \
  --epochs 1 --batch_size 64 --lr 1e-3 --save_dir checkpoints
```

**LoRA (rank=4, alpha=4):**
```bash
python train.py --use_lora --rank 4 --alpha 4 --dropout 0.1 \
  --epochs 1 --batch_size 64 --lr 1e-3 --save_dir checkpoints
```

**LoRA (rank=8, alpha=8):**
```bash
python train.py --use_lora --rank 8 --alpha 8 --dropout 0.1 \
  --epochs 1 --batch_size 64 --lr 1e-3 --save_dir checkpoints
```

**Optuna HPO:**
```bash
python optuna_search.py --n_trials 3
```

**Push best model to HuggingFace:**
```bash
python push_to_hub.py \
  --ckpt checkpoints/LoRA_r4_a4.0_do0.1_best.pt \
  --repo_id Sushantak17/vit-small-lora-cifar100 \
  --rank 4 --alpha 4
```

---

## Q1 Results

### Testing Table

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|------|------|-------|---------|---------------|-----------------|
| No   | -    | -     | -       | 80.96%        | 38,500           |
| Yes  | 2    | 2     | 0.1     | 87.38%        | 93,796           |
| Yes  | 4    | 4     | 0.1     | **87.84%**    | 149,092          |
| Yes  | 8    | 8     | 0.1     | 87.27%        | 259,684          |

**Best config: Rank=4, Alpha=4 → 87.84% test accuracy**

LoRA consistently outperforms baseline (80.96%) across all rank/alpha combinations, achieving 6.88% improvement with only 0.68% of total parameters trainable.

### Train-Val Table — Baseline (10 epochs)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 1.0358    | 0.7642  | 72.58%    | 77.63%  |
| 2     | 0.7083    | 0.7261  | 79.18%    | 78.98%  |
| 3     | 0.6535    | 0.7005  | 80.58%    | 79.61%  |
| 4     | 0.6119    | 0.6979  | 81.58%    | 79.81%  |
| 5     | 0.5745    | 0.6810  | 82.64%    | 80.11%  |
| 6     | 0.5501    | 0.6674  | 83.14%    | 80.70%  |
| 7     | 0.5252    | 0.6665  | 83.91%    | 80.56%  |
| 8     | 0.4969    | 0.6592  | 84.62%    | 80.82%  |
| 9     | 0.4851    | 0.6523  | 85.03%    | 80.92%  |
| 10    | 0.4777    | 0.6485  | 85.31%    | 80.96%  |

### Train-Val Table — LoRA Rank=2, Alpha=2 (1 epoch)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 0.6502    | 0.4096  | 82.12%    | 87.38%  |

### Train-Val Table — LoRA Rank=4, Alpha=4 (1 epoch)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 0.6331    | 0.4058  | 82.48%    | 87.84%  |

### Train-Val Table — LoRA Rank=8, Alpha=8 (1 epoch)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 0.6273    | 0.4079  | 82.53%    | 87.27%  |

### Optuna Results

| Trial | Rank | Alpha | LR      | Val Acc       |
|-------|------|-------|---------|---------------|
| 0     | 4    | 2     | 0.00057 | 75.85%        |
| 1     | 2    | 4     | 0.00082 | 81.00%        |
| 2     | 8    | 2     | 0.00205 | **81.40% ★** |

Optuna best config: Rank=8, Alpha=2, LR=0.00205

---

## Q2: Adversarial Attacks using IBM ART

### Training Commands

**Train ResNet-18 + FGSM attacks:**
```bash
cd Q2
python q2i_fgsm.py --epochs 30 --batch_size 64 --lr 0.1 --save_dir checkpoints_q2i
```

**Train adversarial detectors (PGD + BIM):**
```bash
python q2ii_detector.py \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-3 \
  --n_attack_samples 2000 \
  --victim_ckpt checkpoints_q2i/resnet18_best.pt \
  --save_dir checkpoints_q2ii \
  --pgd_eps 0.1 \
  --pgd_step 0.02 \
  --pgd_iters 20
```

---

## Q2 Results

### Q2(i) — ResNet-18 Clean Accuracy

Final test accuracy: **85.46%**

### Q2(i) — FGSM Attack Comparison

| Attack       | Epsilon | Accuracy | Drop from Clean |
|--------------|---------|----------|-----------------|
| Clean        | -       | 85.90%   | -               |
| FGSM Scratch | 0.01    | 38.40%   | 47.50%          |
| FGSM Scratch | 0.03    | 5.40%    | 80.50%          |
| FGSM Scratch | 0.05    | 2.50%    | 83.40%          |
| FGSM Scratch | 0.10    | 3.60%    | 82.30%          |
| FGSM ART     | 0.01    | 47.60%   | 38.30%          |
| FGSM ART     | 0.03    | 14.30%   | 71.60%          |
| FGSM ART     | 0.05    | 10.30%   | 75.60%          |
| FGSM ART     | 0.10    | 8.70%    | 77.20%          |

### Q2(ii) — Adversarial Detection Results

| Attack | Detector  | Detection Accuracy
|--------|-----------|--------------------|
| PGD    | ResNet-34 | **99.40%**         |
| BIM    | ResNet-34 | **99.40%**         |

---

## Notes

- All experiments run inside Docker container on CPU.
- LoRA experiments run for 1 epoch due to CPU time constraints — results still significantly outperform baseline (80.96% → 87.84%).
- `num_workers=0` used in DataLoader to avoid shared memory errors in Docker.
- Mixed precision (AMP) disabled as CUDA is not available inside Docker on Mac.
- Best Q1 model pushed to HuggingFace: https://huggingface.co/Sushantak17/vit-small-lora-cifar100
