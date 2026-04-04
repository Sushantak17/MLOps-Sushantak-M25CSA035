"""
Q1 Step 5: Optuna Hyperparameter Search for LoRA on CIFAR-100 ViT-S
Fixed for CPU-only Docker on Apple Silicon:
  - num_workers=0, pin_memory=False
  - No autocast / GradScaler
  - pixel_values= keyword for PEFT model
  - 1 epoch per trial, 10% data subset for speed
"""

import os
import argparse
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb
import json


def get_dataloaders(batch_size, subset_fraction=0.1):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR100("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.CIFAR100("data", train=False, download=True, transform=val_tf)

    # Use small subset for speed
    n = int(len(train_ds) * subset_fraction)
    indices = torch.randperm(len(train_ds))[:n].tolist()
    train_ds = Subset(train_ds, indices)

    # Small val subset too
    n_val = int(len(val_ds) * 0.2)
    val_indices = torch.randperm(len(val_ds))[:n_val].tolist()
    val_ds = Subset(val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, val_loader


def build_model(rank, alpha, dropout, device):
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return model.to(device)


def quick_train(model, train_loader, val_loader, lr, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(pixel_values=images).logits
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Final val acc
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(pixel_values=images).logits
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total


def objective(trial, device, wandb_project):
    rank    = trial.suggest_categorical("rank",  [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha", [2, 4, 8])
    lr      = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    dropout = 0.1  # fixed per assignment

    train_loader, val_loader = get_dataloaders(batch_size=64, subset_fraction=0.1)
    model = build_model(rank, alpha, dropout, device)

    val_acc = quick_train(model, train_loader, val_loader, lr,
                          epochs=1, device=device)

    wandb.log({
        "trial":   trial.number,
        "rank":    rank,
        "alpha":   alpha,
        "lr":      lr,
        "val_acc": val_acc,
    })
    print(f"Trial {trial.number}: rank={rank}, alpha={alpha}, "
          f"lr={lr:.5f} → val_acc={val_acc:.2f}%")
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",      type=int, default=3)
    parser.add_argument("--wandb_project", type=str, default="DLOps-Ass5-Q1-Optuna")
    args = parser.parse_args()

    device = torch.device("cpu")

    wandb.init(project=args.wandb_project, name="optuna_sweep")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, device, args.wandb_project),
        n_trials=args.n_trials,
    )

    best = study.best_params
    print("\n=== Best Hyperparameters ===")
    print(json.dumps(best, indent=2))

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/optuna_best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    wandb.log({
        "best_rank":    best["rank"],
        "best_alpha":   best["alpha"],
        "best_lr":      best["lr"],
        "best_val_acc": study.best_value,
    })
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()