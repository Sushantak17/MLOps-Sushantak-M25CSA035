"""
Q2(ii): Adversarial Detection Model using ResNet-34
Fixed for CPU-only Docker on Apple Silicon:
  - No GradScaler, no autocast
  - Stronger attacks (eps=0.1 default)
  - No assert crash
"""

import os
import argparse
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod, FastGradientMethod

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--victim_ckpt",      type=str,   default="checkpoints_q2i/resnet18_best.pt")
    parser.add_argument("--save_dir",         type=str,   default="checkpoints_q2ii")
    parser.add_argument("--wandb_project",    type=str,   default="DLOps-Ass5-Q2ii")
    parser.add_argument("--n_attack_samples", type=int,   default=2000)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--pgd_eps",          type=float, default=0.1)
    parser.add_argument("--pgd_step",         type=float, default=0.02)
    parser.add_argument("--pgd_iters",        type=int,   default=20)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_raw_data(train=True, n=None):
    tf = transforms.ToTensor()
    ds = datasets.CIFAR10("data", train=train, download=True, transform=tf)
    if n and n < len(ds):
        indices = torch.randperm(len(ds))[:n].tolist()
        ds = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_images, all_labels = [], []
    for imgs, lbls in loader:
        all_images.append(imgs)
        all_labels.append(lbls)
    return torch.cat(all_images).numpy(), torch.cat(all_labels).numpy()

def normalize_batch(x_np):
    mean = np.array(CIFAR10_MEAN).reshape(1, 3, 1, 1)
    std  = np.array(CIFAR10_STD).reshape(1, 3, 1, 1)
    return (x_np - mean) / std

def load_victim(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    return model

def build_art_classifier(model):
    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(np.array(CIFAR10_MEAN), np.array(CIFAR10_STD)),
        device_type="cpu",
    )

def generate_pgd(art_clf, x, eps, step, n_iter):
    return ProjectedGradientDescent(
        estimator=art_clf, eps=eps, eps_step=step,
        max_iter=n_iter, norm=np.inf, targeted=False
    ).generate(x=x)

def generate_bim(art_clf, x, eps, step, n_iter):
    return BasicIterativeMethod(
        estimator=art_clf, eps=eps, eps_step=step,
        max_iter=n_iter, targeted=False
    ).generate(x=x)

def build_detector():
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def make_detector_dataset(clean_np, adv_np):
    x = np.concatenate([normalize_batch(clean_np), normalize_batch(adv_np)], axis=0).astype(np.float32)
    y = np.array([0]*len(clean_np) + [1]*len(adv_np), dtype=np.int64)
    idx = np.random.permutation(len(x))
    return TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))

def train_detector(model, train_loader, val_loader, args, device, tag):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc, ckpt = 0.0, os.path.join(args.save_dir, f"detector_{tag}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, leave=False, desc=f"[{tag}] Ep{epoch}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total
        val_loss, val_acc = eval_detector(model, val_loader, criterion, device)
        print(f"[{tag}] Ep {epoch:02d} | TrLoss {train_loss:.4f} TrAcc {train_acc:.2f}% | VaLoss {val_loss:.4f} VaAcc {val_acc:.2f}%")
        wandb.log({f"{tag}/epoch": epoch, f"{tag}/train_loss": train_loss,
                   f"{tag}/train_acc": train_acc, f"{tag}/val_loss": val_loss, f"{tag}/val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)

    print(f"[{tag}] Best: {best_acc:.2f}%")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, best_acc

@torch.no_grad()
def eval_detector(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        loss = criterion(out, lbls)
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total

def log_samples(clean, pgd, bim, fgsm, labels):
    for i in range(min(10, len(clean))):
        lbl = CIFAR10_CLASSES[labels[i]]
        wandb.log({
            f"sample_{i}/clean":    wandb.Image(np.clip(clean[i].transpose(1,2,0),0,1), caption=f"GT:{lbl}"),
            f"sample_{i}/adv_pgd":  wandb.Image(np.clip(pgd[i].transpose(1,2,0),0,1),  caption=f"PGD:{lbl}"),
            f"sample_{i}/adv_bim":  wandb.Image(np.clip(bim[i].transpose(1,2,0),0,1),  caption=f"BIM:{lbl}"),
            f"sample_{i}/adv_fgsm": wandb.Image(np.clip(fgsm[i].transpose(1,2,0),0,1), caption=f"FGSM:{lbl}"),
        })

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    wandb.init(project=args.wandb_project, name="Adversarial_Detector")

    print("Loading victim ResNet-18...")
    victim = load_victim(args.victim_ckpt, device)
    art_clf = build_art_classifier(victim)

    n, n_test = args.n_attack_samples, min(args.n_attack_samples, 1000)
    print(f"Generating adversarial examples (eps={args.pgd_eps}, iters={args.pgd_iters})...")
    x_train_raw, _ = get_raw_data(train=True,  n=n)
    x_test_raw, y_test = get_raw_data(train=False, n=n_test)

    print("  → PGD train..."); x_pgd_train = generate_pgd(art_clf, x_train_raw, args.pgd_eps, args.pgd_step, args.pgd_iters)
    print("  → PGD test...");  x_pgd_test  = generate_pgd(art_clf, x_test_raw,  args.pgd_eps, args.pgd_step, args.pgd_iters)
    print("  → BIM train..."); x_bim_train = generate_bim(art_clf, x_train_raw, args.pgd_eps, args.pgd_step, args.pgd_iters)
    print("  → BIM test...");  x_bim_test  = generate_bim(art_clf, x_test_raw,  args.pgd_eps, args.pgd_step, args.pgd_iters)
    print("  → FGSM vis...");  x_fgsm = FastGradientMethod(estimator=art_clf, eps=args.pgd_eps).generate(x=x_test_raw[:10])

    log_samples(x_test_raw[:10], x_pgd_test[:10], x_bim_test[:10], x_fgsm, y_test[:10])

    # PGD detector
    print("\n─── Training PGD Detector ───")
    pgd_ds = make_detector_dataset(x_train_raw, x_pgd_train)
    n_tr = int(0.8 * len(pgd_ds))
    pgd_tr, pgd_val = torch.utils.data.random_split(pgd_ds, [n_tr, len(pgd_ds)-n_tr])
    pgd_te_ds = make_detector_dataset(x_test_raw, x_pgd_test)
    pgd_det = build_detector().to(device)
    pgd_det, _ = train_detector(pgd_det,
        DataLoader(pgd_tr,    args.batch_size, shuffle=True,  num_workers=2),
        DataLoader(pgd_val,   args.batch_size, shuffle=False, num_workers=2),
        args, device, "PGD")
    _, pgd_acc = eval_detector(pgd_det, DataLoader(pgd_te_ds, args.batch_size, shuffle=False), nn.CrossEntropyLoss(), device)

    # BIM detector
    print("\n─── Training BIM Detector ───")
    bim_ds = make_detector_dataset(x_train_raw, x_bim_train)
    n_tr = int(0.8 * len(bim_ds))
    bim_tr, bim_val = torch.utils.data.random_split(bim_ds, [n_tr, len(bim_ds)-n_tr])
    bim_te_ds = make_detector_dataset(x_test_raw, x_bim_test)
    bim_det = build_detector().to(device)
    bim_det, _ = train_detector(bim_det,
        DataLoader(bim_tr,    args.batch_size, shuffle=True,  num_workers=2),
        DataLoader(bim_val,   args.batch_size, shuffle=False, num_workers=2),
        args, device, "BIM")
    _, bim_acc = eval_detector(bim_det, DataLoader(bim_te_ds, args.batch_size, shuffle=False), nn.CrossEntropyLoss(), device)

    print(f"\n===== Summary =====")
    print(f"PGD Detection Accuracy: {pgd_acc:.2f}%")
    print(f"BIM Detection Accuracy: {bim_acc:.2f}%")
    wandb.log({"PGD_final_detection_acc": pgd_acc, "BIM_final_detection_acc": bim_acc})
    wandb.finish()
    print("Done.")

if __name__ == "__main__":
    main()