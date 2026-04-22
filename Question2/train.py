import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 23
IMG_H, IMG_W = 128, 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

DATA_ROOT = "./dataset/data"
RGB_DIR = os.path.join(DATA_ROOT, "CameraRGB")
MASK_DIR = os.path.join(DATA_ROOT, "CameraMask")


class CityscapeDataset(Dataset):
    def __init__(self, file_list, rgb_dir, mask_dir, img_size=(IMG_H, IMG_W)):
        self.file_list = file_list
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = Image.open(os.path.join(self.rgb_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname))
        img = self.img_tf(img)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        mask = torch.from_numpy(mask).long()
        return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_c=3, n_classes=NUM_CLASSES, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        prev = in_c
        for f in features:
            self.downs.append(DoubleConv(prev, f))
            prev = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


def compute_metrics(preds, targets, n_classes=NUM_CLASSES):
    preds = preds.view(-1)
    targets = targets.view(-1)
    ious, dices = [], []
    for c in range(n_classes):
        p = (preds == c)
        t = (targets == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()
        if t_sum == 0 and p_sum == 0:
            continue
        iou = inter / (union + 1e-8)
        dice = (2 * inter) / (p_sum + t_sum + 1e-8)
        ious.append(iou)
        dices.append(dice)
    m_iou = np.mean(ious) if ious else 0.0
    m_dice = np.mean(dices) if dices else 0.0
    return m_iou, m_dice


def evaluate(model, loader):
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            miou, mdice = compute_metrics(preds, masks)
            ious.append(miou)
            dices.append(mdice)
    return np.mean(ious), np.mean(dices)


def main():
    files = sorted([f for f in os.listdir(RGB_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Total samples: {len(files)}")

    train_files, test_files = train_test_split(files, test_size=0.2, random_state=SEED)
    print(f"Train: {len(train_files)}  Test: {len(test_files)}")

    with open("test_files.json", "w") as f:
        json.dump(test_files, f)

    train_ds = CityscapeDataset(train_files, RGB_DIR, MASK_DIR)
    test_ds = CityscapeDataset(test_files, RGB_DIR, MASK_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, train_mious, train_mdices = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_iou, running_dice = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = torch.argmax(out, dim=1)
                miou, mdice = compute_metrics(preds, masks)
            running_loss += loss.item()
            running_iou += miou
            running_dice += mdice
            pbar.set_postfix(loss=loss.item(), miou=miou, mdice=mdice)

        n = len(train_loader)
        avg_loss = running_loss / n
        avg_iou = running_iou / n
        avg_dice = running_dice / n
        train_losses.append(avg_loss)
        train_mious.append(avg_iou)
        train_mdices.append(avg_dice)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  mIoU={avg_iou:.4f}  mDice={avg_dice:.4f}")

    torch.save(model.state_dict(), "models/unet_cityscape.pth")
    print("Model saved.")

    test_miou, test_mdice = evaluate(model, test_loader)
    print(f"\nTEST mIoU: {test_miou:.4f}")
    print(f"TEST mDice: {test_mdice:.4f}")

    with open("test_metrics.json", "w") as f:
        json.dump({"mIoU": float(test_miou), "mDice": float(test_mdice)}, f)

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_losses, marker="o", color="red")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_mious, marker="o", color="blue")
    plt.title("Training mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.grid(True)
    plt.savefig("plots/miou_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_mdices, marker="o", color="green")
    plt.title("Training mDice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.grid(True)
    plt.savefig("plots/mdice_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    with open("train_history.json", "w") as f:
        json.dump({
            "loss": train_losses,
            "mIoU": train_mious,
            "mDice": train_mdices,
        }, f)

    print("All plots saved in plots/")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    main()
