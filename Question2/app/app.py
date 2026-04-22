import os
import json
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms

NUM_CLASSES = 23
IMG_H, IMG_W = 128, 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT, "plots")
MODEL_PATH = os.path.join(ROOT, "models", "unet_cityscape.pth")
METRICS_PATH = os.path.join(ROOT, "test_metrics.json")
MASK_DIR = os.path.join(ROOT, "dataset", "data", "CameraMask")


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


@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess(img_pil):
    tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])
    return tf(img_pil.convert("RGB")).unsqueeze(0)


def colorize_mask(mask, n_classes=NUM_CLASSES):
    np.random.seed(42)
    colors = np.random.randint(0, 255, (n_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        out[mask == c] = colors[c]
    return out


def predict(model, img_pil):
    x = preprocess(img_pil).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred


st.set_page_config(page_title="Cityscape Segmentation", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training Results", "Predict on Images"])

if page == "Training Results":
    st.title("Cityscape Segmentation - Training Results")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("Test mIoU", f"{m['mIoU']:.4f}")
        c2.metric("Test mDice", f"{m['mDice']:.4f}")
    else:
        st.warning("Test metrics file not found.")

    st.subheader("Training Plots")
    plot_files = [
        ("Training Loss", "loss_curve.png"),
        ("Training mIoU", "miou_curve.png"),
        ("Training mDice", "mdice_curve.png"),
    ]
    cols = st.columns(3)
    for i, (title, fname) in enumerate(plot_files):
        path = os.path.join(PLOTS_DIR, fname)
        with cols[i]:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"{fname} not found")

else:
    st.title("Predict on Test Images")
    st.write("Upload exactly 4 images from the test set. The app shows input, ground-truth mask, and predicted mask.")

    uploads = st.file_uploader(
        "Upload 4 input images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploads:
        if len(uploads) != 4:
            st.warning(f"Please upload exactly 4 images. You uploaded {len(uploads)}.")
        else:
            model = load_model()
            for up in uploads:
                st.markdown(f"### {up.name}")
                img = Image.open(up).convert("RGB")
                pred = predict(model, img)
                pred_rgb = colorize_mask(pred)

                gt_path = os.path.join(MASK_DIR, up.name)
                gt_rgb = None
                if os.path.exists(gt_path):
                    gt = Image.open(gt_path).resize((IMG_W, IMG_H), Image.NEAREST)
                    gt_np = np.array(gt)
                    if gt_np.ndim == 3:
                        gt_np = gt_np[:, :, 0]
                    gt_np = np.clip(gt_np, 0, NUM_CLASSES - 1)
                    gt_rgb = colorize_mask(gt_np)

                c1, c2, c3 = st.columns(3)
                c1.image(img.resize((IMG_W, IMG_H)), caption="Input", use_container_width=True)
                if gt_rgb is not None:
                    c2.image(gt_rgb, caption="Ground Truth Mask", use_container_width=True)
                else:
                    c2.info("GT mask not found")
                c3.image(pred_rgb, caption="Predicted Mask", use_container_width=True)
