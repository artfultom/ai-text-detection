import logging
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_text_detector.inference.dto import ScoreResponse, ScoreRequest, TextScore
from ai_text_detector.inference.envs import THRESHOLD, MODEL_DIR, PORT, DEVICE
from ai_text_detector.inference.service import build_feature, score_features
from ai_text_detector.inference.state import AppState
from ai_text_detector.train.models import load_model_and_encoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading model...")
    ae, encoder, metadata = load_model_and_encoder()
    state.ae = ae
    state.encoder = encoder
    state.metadata = metadata
    state.device = DEVICE

    log.info("Ready on port %d", PORT)
    yield


app = FastAPI(
    title="AI Text Detector",
    description="Scores texts for AI origin using a trained Autoencoder.",
    version="1.0.0",
    lifespan=lifespan,
)


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
        X = np.array([build_feature(t, state) for t in req.texts])
        scores = score_features(X, state)
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
