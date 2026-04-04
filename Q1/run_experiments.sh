#!/bin/bash
# Run all Q1 experiments inside Docker
# Usage: bash run_experiments.sh

set -e

SAVE_DIR="checkpoints"
mkdir -p $SAVE_DIR

echo "============================================"
echo "  Q1a: Baseline (Head-only fine-tuning)"
echo "============================================"
python train.py \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir $SAVE_DIR

echo ""
echo "============================================"
echo "  Q1b: LoRA experiments (rank x alpha)"
echo "============================================"

RANKS=(2 4 8)
ALPHAS=(2 4 8)

for rank in "${RANKS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        echo "  → rank=$rank alpha=$alpha"
        python train.py \
            --use_lora \
            --rank $rank \
            --alpha $alpha \
            --dropout 0.1 \
            --epochs 10 \
            --batch_size 128 \
            --lr 1e-3 \
            --save_dir $SAVE_DIR
    done
done

echo ""
echo "============================================"
echo "  Q1 Step 5: Optuna HPO"
echo "============================================"
python optuna_search.py --n_trials 20

echo ""
echo "============================================"
echo "  Q1 Optional: Partial Freeze + LoRA"
echo "============================================"
# Update rank/alpha to best found by Optuna (edit below after running HPO)
python train.py \
    --use_lora \
    --partial_freeze \
    --rank 4 \
    --alpha 8 \
    --dropout 0.1 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir $SAVE_DIR

echo "All Q1 experiments done!"
