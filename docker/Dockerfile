FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p data Q1/checkpoints Q2/checkpoints_q2i Q2/checkpoints_q2ii

CMD ["bash"]