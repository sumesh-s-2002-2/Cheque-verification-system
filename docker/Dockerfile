FROM python:3.11-slim

#dependencies needed for opencv 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libffi-dev \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p \
    src/data/raw \
    src/data/synthetic \
    src/data/processed \
    src/data/signatures \
    src/data/labels \
    src/models/checkpoints \
    mlruns

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "src/main.py"]
