"""
Q2(i): ResNet-18 on CIFAR-10, trained from scratch, then attacked with:
  - FGSM (custom implementation)
  - FGSM via IBM ART
Includes visual comparison and WandB logging.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# IBM ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Q2(i) FGSM on CIFAR-10")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints_q2i")
    parser.add_argument("--wandb_project",type=str,   default="DLOps-Ass5-Q2i")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_dataloaders(batch_size):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10("data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def get_raw_test_data(n=1000):
    """Return raw (unnormalized) test images in [0,1] and labels for ART."""
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler()
    best_acc = 0.0
    ckpt_path = os.path.join(args.save_dir, "resnet18_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, leave=False, desc=f"Ep{epoch}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
        scheduler.step()

        train_acc  = 100.0 * correct / total
        train_loss = total_loss / total
        val_loss, val_acc = evaluate_loader(model, test_loader, criterion, device)
        print(f"Epoch {epoch:02d} | TrLoss {train_loss:.4f} TrAcc {train_acc:.2f}% "
              f"| VaLoss {val_loss:.4f} VaAcc {val_acc:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"Best val accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.amp.autocast("cuda"):
            out = model(imgs)
            loss = criterion(out, lbls)
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# FGSM from Scratch
# ──────────────────────────────────────────────────────────────────────────────

def normalize_tensor(x, mean, std, device):
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    return (x - m) / s


def fgsm_scratch(model, images_raw, labels, epsilon, device):
    """
    images_raw: [B, C, H, W] in [0,1], unnormalized
    Returns adversarial images in [0,1], unnormalized
    """
    model.eval()
    x = images_raw.clone().to(device)
    x.requires_grad_(True)

    x_norm = normalize_tensor(x, CIFAR10_MEAN, CIFAR10_STD, device)
    out = model(x_norm)
    loss = nn.CrossEntropyLoss()(out, labels.to(device))
    model.zero_grad()
    loss.backward()

    grad_sign = x.grad.data.sign()
    x_adv = (x + epsilon * grad_sign).clamp(0, 1).detach()
    return x_adv


def eval_on_raw(model, images_raw, labels, device):
    model.eval()
    x_norm = normalize_tensor(
        images_raw.to(device), CIFAR10_MEAN, CIFAR10_STD, device
    )
    with torch.no_grad():
        out = model(x_norm)
    preds = out.argmax(1)
    acc = 100.0 * preds.eq(labels.to(device)).float().mean().item()
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


def save_comparison_grid(clean, adv_scratch, adv_art, labels, preds_clean,
                         preds_scratch, preds_art, n=10, path="comparison.png"):
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    titles = ["Clean", "FGSM Scratch", "FGSM ART"]
    all_imgs = [clean, adv_scratch, adv_art]
    all_preds = [preds_clean, preds_scratch, preds_art]

    for row, (imgs, preds, title) in enumerate(zip(all_imgs, all_preds, titles)):
        for col in range(n):
            img = imgs[col]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            pred_lbl = CIFAR10_CLASSES[int(preds[col])]
            true_lbl = CIFAR10_CLASSES[int(labels[col])]
            color = "green" if int(preds[col]) == int(labels[col]) else "red"
            ax.set_title(f"GT:{true_lbl}\nP:{pred_lbl}", fontsize=6, color=color)
        axes[row][0].set_ylabel(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_eps_vs_accuracy(epsilons, acc_scratch, acc_art, path):
    fig, ax = plt.subplots()
    ax.plot(epsilons, acc_scratch, "o-", label="FGSM Scratch")
    ax.plot(epsilons, acc_art,     "s-", label="FGSM ART")
    ax.set_xlabel("Epsilon (Perturbation Strength)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Perturbation Strength vs Accuracy Drop")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, name="FGSM_comparison")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = build_resnet18().to(device)

    # ── 1. Train from scratch ──────────────────────────────────────────────
    print("Training ResNet-18 from scratch on CIFAR-10...")
    model = train(model, train_loader, test_loader, args, device)

    # Evaluate clean
    criterion = nn.CrossEntropyLoss()
    _, clean_acc = evaluate_loader(model, test_loader, criterion, device)
    print(f"Clean test accuracy: {clean_acc:.2f}%")
    assert clean_acc >= 72.0, f"Clean acc {clean_acc:.2f}% < 72%. Consider training longer."

    # ── 2 & 3. FGSM attacks ───────────────────────────────────────────────
    images_raw, labels_np = get_raw_test_data(n=1000)
    images_t  = torch.tensor(images_raw)
    labels_t  = torch.tensor(labels_np)

    # Build ART classifier (operates on raw [0,1] images with preprocessing)
    art_model = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(np.array(CIFAR10_MEAN), np.array(CIFAR10_STD)),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    epsilons = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    acc_clean_on_raw = eval_on_raw(model, images_t, labels_t, device)
    print(f"Clean acc (raw subset): {acc_clean_on_raw:.2f}%")

    acc_scratch_list, acc_art_list = [], []

    for eps in epsilons:
        # FGSM Scratch
        adv_scratch = fgsm_scratch(model, images_t, labels_t, eps, device)
        acc_s = eval_on_raw(model, adv_scratch, labels_t, device)
        acc_scratch_list.append(acc_s)

        # FGSM ART
        fgsm_art = FastGradientMethod(estimator=art_model, eps=eps)
        adv_art_np = fgsm_art.generate(x=images_raw)
        adv_art_t  = torch.tensor(adv_art_np)
        acc_a = eval_on_raw(model, adv_art_t, labels_t, device)
        acc_art_list.append(acc_a)

        print(f"eps={eps:.3f} | Scratch acc={acc_s:.2f}% | ART acc={acc_a:.2f}%")
        wandb.log({f"fgsm_scratch_acc_eps{eps}": acc_s,
                   f"fgsm_art_acc_eps{eps}": acc_a, "epsilon": eps})

    # Save perturbation vs accuracy plot
    plot_path = os.path.join(args.save_dir, "eps_vs_accuracy.png")
    plot_eps_vs_accuracy(epsilons, acc_scratch_list, acc_art_list, plot_path)
    wandb.log({"perturbation_vs_accuracy": wandb.Image(plot_path)})

    # ── 4. Visual comparison (eps=0.03) ───────────────────────────────────
    eps_vis = 0.03
    adv_s_vis = fgsm_scratch(model, images_t[:10], labels_t[:10], eps_vis, device)

    fgsm_art = FastGradientMethod(estimator=art_model, eps=eps_vis)
    adv_a_vis = torch.tensor(fgsm_art.generate(x=images_raw[:10]))

    # Get predictions
    def get_preds_raw(imgs):
        model.eval()
        x = normalize_tensor(imgs.to(device), CIFAR10_MEAN, CIFAR10_STD, device)
        with torch.no_grad():
            return model(x).argmax(1).cpu().numpy()

    preds_clean   = get_preds_raw(images_t[:10])
    preds_scratch = get_preds_raw(adv_s_vis.cpu())
    preds_art     = get_preds_raw(adv_a_vis)

    comp_path = os.path.join(args.save_dir, "fgsm_comparison.png")
    save_comparison_grid(
        images_t[:10], adv_s_vis.cpu(), adv_a_vis,
        labels_t[:10], preds_clean, preds_scratch, preds_art,
        n=10, path=comp_path
    )
    wandb.log({"fgsm_visual_comparison": wandb.Image(comp_path)})

    # ── WandB: log 10 samples each for clean + adv ────────────────────────
    for i in range(10):
        wandb.log({
            "samples/clean": wandb.Image(
                images_t[i].permute(1, 2, 0).numpy(),
                caption=f"GT:{CIFAR10_CLASSES[labels_np[i]]} Pred:{CIFAR10_CLASSES[preds_clean[i]]}"
            ),
            "samples/fgsm_scratch": wandb.Image(
                adv_s_vis[i].cpu().permute(1, 2, 0).numpy(),
                caption=f"GT:{CIFAR10_CLASSES[labels_np[i]]} Pred:{CIFAR10_CLASSES[preds_scratch[i]]}"
            ),
            "samples/fgsm_art": wandb.Image(
                adv_a_vis[i].permute(1, 2, 0).numpy(),
                caption=f"GT:{CIFAR10_CLASSES[labels_np[i]]} Pred:{CIFAR10_CLASSES[preds_art[i]]}"
            ),
        })

    # ── 5 & 6. Summary ────────────────────────────────────────────────────
    print("\n===== FGSM Summary =====")
    print(f"Clean acc:           {acc_clean_on_raw:.2f}%")
    for eps, acc_s, acc_a in zip(epsilons, acc_scratch_list, acc_art_list):
        drop_s = acc_clean_on_raw - acc_s
        drop_a = acc_clean_on_raw - acc_a
        print(f"eps={eps:.2f} | Scratch: {acc_s:.2f}% (drop {drop_s:.2f}%) "
              f"| ART: {acc_a:.2f}% (drop {drop_a:.2f}%)")

    wandb.finish()


if __name__ == "__main__":
    main()
