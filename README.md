# Question2: CityScape Image Segmentation

## Overview
End-to-end UNet-based image segmentation pipeline deployed using Streamlit/Gradio.

## Dataset
- RGB Images: `data/CameraRGB/`
- Masks: `data/CameraMask/`
- Classes: 23  
- Split: 80% Train / 20% Test (seed = 42)

## Model
- UNet architecture  
- Trained for 15+ epochs  

## Results
- mIOU: 0.5472  
- mDICE: 0.6125  

## App Features
- Page 1: Training loss, mIOU, mDice plots + test scores  
- Page 2: Upload 4 images → displays ground truth and predicted masks  

## Note
Training plots and app screenshots are included in this repository.
