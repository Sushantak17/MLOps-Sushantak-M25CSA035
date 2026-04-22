FROM python:3.6-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translate.py .
COPY evaluate.py .
COPY input.txt .
COPY reference.txt .

CMD ["sh", "-c", "python translate.py && python evaluate.py"]