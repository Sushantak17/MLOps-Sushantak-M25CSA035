# Question2 - Cityscape Image Segmentation

UNet-based semantic segmentation on the Cityscape dataset (23 classes).

## Results

Question2: mIOU: 0.5472 and mDICE: 0.6125

## Setup
- Model: UNet (encoder 64-128-256-512, bottleneck 1024)
- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=1e-3
- Batch size: 16
- Image size: 128x128
- Epochs: 20
- Train/Test split: 80/20, seed=42
- Total samples: 1060 (Train: 848, Test: 212)

## Files
- train.py - training script
- app/app.py - Streamlit 2-page app
- plots/ - training curves (loss, mIoU, mDice)
- models/ - trained weights
- test_metrics.json - final test metrics

## Run
python train.py
streamlit run app/app.py
