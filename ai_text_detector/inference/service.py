import re

import numpy as np
import torch

from ai_text_detector.train.embeddings import compute_poolings


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())

    return [p for p in parts if p] or [text]


def build_feature(text: str, state) -> np.ndarray:
    sentences = split_sentences(text)
    emb = state.encoder.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    mean_val, mean_diff_val = compute_poolings(emb)

    return np.concatenate([mean_val * 1.0, mean_diff_val * 1.2]).astype(np.float32)


def score_features(X: np.ndarray, state) -> np.ndarray:
    X_t = torch.tensor(X).to(state.device)
    with torch.no_grad():
        recon = state.ae(X_t).cpu().numpy()

    return np.mean((X - recon) ** 2, axis=1)
