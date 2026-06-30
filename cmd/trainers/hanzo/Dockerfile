FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# build-essential + git for bitsandbytes / source installs.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install the trainer's Python dependencies first (layer-cached).
COPY cmd/trainers/hanzo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The env-driven fine-tuning entrypoint (LoRA / QLoRA / full).
COPY cmd/trainers/hanzo/finetune.py /app/finetune.py

ENTRYPOINT ["python", "/app/finetune.py"]
