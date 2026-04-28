import glob
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download, list_repo_files
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/train_results")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
ENCODER_NAME = os.environ.get("ENCODER_NAME", "")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", "8000"))
THRESHOLD = float(os.environ["THRESHOLD"]) if "THRESHOLD" in os.environ else None


class AE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class AppState:
    ae: AE
    encoder: SentenceTransformer
    metadata: dict
    device: str


state = AppState()


def _find_local_checkpoint() -> Optional[str]:
    pattern = os.path.join(MODEL_DIR, "ae_*.pt")
    matches = sorted(glob.glob(pattern))

    return matches[0] if matches else None


def _download_from_hf() -> str:
    if not HF_REPO_ID:
        raise RuntimeError(
            "No .pt checkpoint found locally and HF_REPO_ID is not set. "
            "Either mount a checkpoint into MODEL_DIR or set HF_REPO_ID."
        )

    log.info("Searching HuggingFace repo: %s", HF_REPO_ID)
    repo_files = list(list_repo_files(HF_REPO_ID, token=HF_TOKEN))
    pt_files = [f for f in repo_files if f.startswith("ae_") and f.endswith(".pt")]

    if not pt_files:
        raise RuntimeError(f"No ae_*.pt file found in HuggingFace repo {HF_REPO_ID}")

    filename = pt_files[0]
    log.info("Downloading checkpoint: %s", filename)
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN,
        local_dir=MODEL_DIR,
    )
    log.info("Saved to: %s", local_path)

    return local_path


def load_model_and_encoder() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_path = _find_local_checkpoint()
    if ckpt_path:
        log.info("Found local checkpoint: %s", ckpt_path)
    else:
        log.info("No local checkpoint in %s — downloading from HuggingFace...", MODEL_DIR)
        ckpt_path = _download_from_hf()

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    metadata: dict = checkpoint.get("metadata", {})
    state_dict = checkpoint["state_dict"]

    dim = state_dict["enc.0.weight"].shape[1]
    ae = AE(dim).to(DEVICE)
    ae.load_state_dict(state_dict)
    ae.eval()
    log.info("AE loaded  input_dim=%d  metadata=%s", dim, metadata)

    encoder_name = ENCODER_NAME or metadata.get("model")
    if not encoder_name:
        raise RuntimeError(
            "Cannot determine encoder name. "
            "Set ENCODER_NAME env var or ensure checkpoint metadata contains 'model'."
        )
    log.info("Loading sentence encoder: %s", encoder_name)
    encoder = SentenceTransformer(encoder_name, device=DEVICE)

    state.ae = ae
    state.encoder = encoder
    state.metadata = metadata
    state.device = DEVICE


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(v).float()

    return F.normalize(t, p=2, dim=0).numpy()


def _compute_poolings(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_val = np.mean(emb, axis=0)
    diffs = emb[1:] - emb[:-1]
    mean_diff_val = np.mean(diffs, axis=0) if len(diffs) else np.zeros_like(mean_val)

    return mean_val, _normalize_vec(mean_diff_val)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())

    return [p for p in parts if p] or [text]


def _build_feature(text: str) -> np.ndarray:
    sentences = _split_sentences(text)
    emb = state.encoder.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    mean_val, mean_diff_val = _compute_poolings(emb)

    return np.concatenate([mean_val * 1.0, mean_diff_val * 1.2]).astype(np.float32)


def _score_features(X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X).to(state.device)
    with torch.no_grad():
        recon = state.ae(X_t).cpu().numpy()

    return np.mean((X - recon) ** 2, axis=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading model...")
    load_model_and_encoder()
    log.info("Ready on port %d", PORT)
    yield


app = FastAPI(
    title="AI Text Detector",
    description="Scores texts for AI origin using a trained Autoencoder.",
    version="1.0.0",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to score")
    threshold: Optional[float] = Field(
        default=None,
        description="Decision threshold. If set, returns 'AI'/'Human' labels.",
    )


class TextScore(BaseModel):
    text: str
    score: float
    label: Optional[str] = None


class ScoreResponse(BaseModel):
    results: list[TextScore]
    model: str
    threshold: Optional[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "metadata": state.metadata,
        "device": state.device,
        "model_dir": MODEL_DIR,
        "default_threshold": THRESHOLD,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list is empty")

    try:
        X = np.array([_build_feature(t) for t in req.texts])
        scores = _score_features(X)
    except Exception as exc:
        log.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(exc))

    threshold = req.threshold if req.threshold is not None else THRESHOLD

    results = []
    for text, score_val in zip(req.texts, scores.tolist()):
        label = None
        if threshold is not None:
            label = "AI" if score_val >= threshold else "Human"
        results.append(TextScore(text=text, score=round(score_val, 8), label=label))

    return ScoreResponse(
        results=results,
        model=state.metadata.get("model", "unknown"),
        threshold=threshold,
    )


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=PORT, reload=False)
