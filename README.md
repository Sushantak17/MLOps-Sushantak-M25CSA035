# ResNet Docker Training (MLOps Assignment)

## Project Overview
Training ResNet18 using Docker container with GPU acceleration.

---

## Folder Structure

resnet_docker/
│
├── train.py
├── Dockerfile
├── README.md
└── data/train/

---

## Build Docker Image

```bash
docker build -t resnet-training:v1 .

## Running container

docker run -it --rm \
--gpus all \
--shm-size=8g \
-v $(pwd):/workspace \
resnet-training:v1
