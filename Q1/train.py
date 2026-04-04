"""
Q1: ViT-S Fine-tuning with LoRA on CIFAR-100
Supports:
  - Baseline (head-only fine-tuning)
  - LoRA with various rank/alpha combinations via PEFT
  - WandB logging
  - Gradient update graphs for LoRA weights

Fixed for CPU-only Docker on Apple Silicon:
  - num_workers=0 (fixes shared memory error)
  - No autocast / GradScaler (no CUDA)
  - pin_memory=False
"""

import os
import argparse
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Config & Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ViT-S LoRA on CIFAR-100")
    parser.add_argument("--use_lora",       action="store_true", help="Use LoRA")
    parser.add_argument("--rank",           type=int,   default=4, choices=[2, 4, 8])
    parser.add_argument("--alpha",          type=float, default=4, choices=[2, 4, 8])
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--num_classes",    type=int,   default=100)
    parser.add_argument("--save_dir",       type=str,   default="checkpoints")
    parser.add_argument("--wandb_project",  type=str,   default="DLOps-Ass5-Q1")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num_workers",    type=int,   default=0,
                        help="Set to 0 for Docker/CPU environments")
    parser.add_argument("--partial_freeze", action="store_true",
                        help="Freeze first half of ViT blocks, LoRA on frozen part")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size, num_workers=0):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=28),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100(root="data", train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR100(root="data", train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args, device):
    model_name = "WinKawaks/vit-small-patch16-224"

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
    )

    if args.use_lora:
        target_modules = ["query", "key", "value"]
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

        if args.partial_freeze:
            for i in range(6):
                block_prefix = f"base_model.model.vit.encoder.layer.{i}."
                for name, param in model.named_parameters():
                    if name.startswith(block_prefix) and "lora_" not in name:
                        param.requires_grad = False

        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model.to(device)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Training / Validation
# ──────────────────────────────────────────────────────────────────────────────

def model_forward(model, images):
    """Handle both normal ViT and PEFT-wrapped ViT forward pass."""
    try:
        return model(pixel_values=images).logits
    except TypeError:
        return model(images).logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_forward(model, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, leave=False, desc="Val"):
        images, labels = images.to(device), labels.to(device)
        outputs = model_forward(model, images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Gradient Tracking
# ──────────────────────────────────────────────────────────────────────────────

class GradientTracker:
    def __init__(self, model):
        self.hooks = []
        self.grad_norms = {}
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.grad_norms[name] = []
                self.hooks.append(param.register_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(grad):
            self.grad_norms[name].append(grad.norm().item())
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

    def get_epoch_stats(self):
        stats = {}
        for name, norms in self.grad_norms.items():
            if norms:
                stats[name] = np.mean(norms)
                self.grad_norms[name] = []
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_classwise_accuracy(preds, labels, num_classes, save_path):
    per_class_correct = np.zeros(num_classes)
    per_class_total   = np.zeros(num_classes)
    for p, l in zip(preds, labels):
        per_class_total[l] += 1
        if p == l:
            per_class_correct[l] += 1
    acc = 100.0 * per_class_correct / (per_class_total + 1e-8)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(num_classes), acc)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Class-wise Test Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return acc


def plot_lora_gradients(grad_history, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, norms in grad_history.items():
        short = name.split(".")[-3] + "." + name.split(".")[-1]
        ax.plot(norms, label=short)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Gradient Norm")
    ax.set_title("LoRA Weight Gradient Updates During Training")
    ax.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")  # CPU only in Docker on Mac

    run_name = (
        f"LoRA_r{args.rank}_a{args.alpha}_do{args.dropout}"
        if args.use_lora else "Baseline_HeadOnly"
    )
    if args.partial_freeze:
        run_name += "_PartialFreeze"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader = get_dataloaders(args.batch_size, args.num_workers)
    model = build_model(args, device)
    trainable = count_trainable_params(model)
    print(f"Trainable parameters: {trainable:,}")
    wandb.config.update({"trainable_params": trainable})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    grad_tracker = GradientTracker(model) if args.use_lora else None
    grad_epoch_history = {}

    results_table = []
    best_val_acc = 0.0
    best_ckpt = os.path.join(args.save_dir, f"{run_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"TrLoss {train_loss:.4f} | TrAcc {train_acc:.2f}% | "
              f"VaLoss {val_loss:.4f} | VaAcc {val_acc:.2f}%")

        log_dict = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
        }

        if grad_tracker:
            stats = grad_tracker.get_epoch_stats()
            for name, norm in stats.items():
                log_dict[f"grad_norm/{name}"] = norm
                grad_epoch_history.setdefault(name, []).append(norm)

        wandb.log(log_dict)
        results_table.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_acc":    round(val_acc, 2),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

    # ── Final test evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_acc, test_preds, test_labels = evaluate(model, val_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    hist_path = os.path.join(args.save_dir, f"{run_name}_classwise.png")
    plot_classwise_accuracy(test_preds, test_labels, args.num_classes, hist_path)
    wandb.log({"classwise_accuracy": wandb.Image(hist_path)})

    if grad_epoch_history:
        grad_path = os.path.join(args.save_dir, f"{run_name}_lora_grads.png")
        plot_lora_gradients(grad_epoch_history, grad_path)
        wandb.log({"lora_gradient_updates": wandb.Image(grad_path)})
        grad_tracker.remove()

    summary = {
        "run_name":        run_name,
        "use_lora":        args.use_lora,
        "rank":            args.rank if args.use_lora else None,
        "alpha":           args.alpha if args.use_lora else None,
        "dropout":         args.dropout if args.use_lora else None,
        "trainable_params": trainable,
        "best_val_acc":    round(best_val_acc, 2),
        "test_acc":        round(test_acc, 2),
        "epoch_results":   results_table,
    }
    with open(os.path.join(args.save_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    wandb.log({
        "best_val_acc":    best_val_acc,
        "test_acc":        test_acc,
        "trainable_params": trainable,
    })
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()