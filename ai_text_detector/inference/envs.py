import os

import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/train_results")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
ENCODER_NAME = os.environ.get("ENCODER_NAME", "")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", "8080"))
THRESHOLD = float(os.environ["THRESHOLD"]) if "THRESHOLD" in os.environ else None
