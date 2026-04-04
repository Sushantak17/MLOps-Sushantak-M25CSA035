"""
Push best LoRA ViT-S model to HuggingFace Hub.
Usage:
    python push_to_hub.py \
        --ckpt checkpoints/LoRA_r4_a8_do0.1_best.pt \
        --repo_id YOUR_USERNAME/vit-small-lora-cifar100 \
        --rank 4 --alpha 8 --dropout 0.1
"""

import argparse
import json
import torch
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--repo_id",  required=True)
    parser.add_argument("--rank",     type=int,   default=4)
    parser.add_argument("--alpha",    type=float, default=8)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--hf_token", type=str,   default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)

    print(f"Pushing to {args.repo_id} ...")
    model.push_to_hub(args.repo_id, token=args.hf_token)
    print("Done.")


if __name__ == "__main__":
    main()
