import glob
import hashlib
import logging
import os
import re

import hydra
import nltk
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ai_text_detector.utils.logging import log, section

logging.getLogger("httpx").setLevel(logging.WARNING)


def load_essays(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    log(f"Loaded {len(df):,} rows from {os.path.basename(path)}", 1)

    return df


def _compute_hash(data, model_name):
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))

    for text in data:
        h.update(text.encode("utf-8"))

    return h.hexdigest()[:32]


def count_rows_by_pattern(input_dir: str, patterns: list[str]) -> int:
    total = 0
    for pattern in patterns:
        paths = glob.glob(os.path.join(input_dir, pattern))
        if not paths:
            raise ValueError(f"No files matched pattern: {pattern}")

        count = sum(len(pd.read_csv(p)) for p in paths)
        log(f"{pattern} → {len(paths)} files, {count:,} rows", 2)
        total += count

    log(f"Total matched rows: {total:,}", 2)

    return total


def load_source(cfg, source_cfg, input_dir: str) -> pd.DataFrame:
    dfs = []

    for file_cfg in source_cfg:
        input_path = os.path.join(input_dir, file_cfg.path)
        df = load_essays(input_path)

        if "n_from_pattern" in file_cfg:
            patterns = list(file_cfg.n_from_pattern)
            n = count_rows_by_pattern(input_dir, patterns)

            df = df.sample(n=min(n, len(df)), random_state=cfg.run.seed)
            log(f"Sampled {len(df):,} rows", 1)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    log(f"Final dataset size: {len(df):,}", 1)

    return df


def compute_embeddings(
        data: list[str],
        model: SentenceTransformer,
        model_name: str,
        device: str,
        embed_cfg,
):
    all_sentences = []
    sentence_counts = []

    for text in tqdm(data, desc="Sentences", leave=False):
        text = re.sub(r"\n+", ". ", text)
        sentences = nltk.sent_tokenize(text)

        if "e5" in model_name:
            sentences = [f"passage: {s}" for s in sentences]

        all_sentences.extend(sentences)
        sentence_counts.append(len(sentences))

    log(f"Total sentences: {len(all_sentences):,}", 1)

    log(f"Encoding on {device} (batch={embed_cfg.batch_size})...", 1)

    embeddings = model.encode(
        all_sentences,
        batch_size=embed_cfg.batch_size,
        normalize_embeddings=False,
        show_progress_bar=True,
        device=device,
    )

    return embeddings, sentence_counts


def save_embeddings(
        texts,
        all_embeddings: np.ndarray,
        sentence_counts: list[int],
        embed_cfg,
        model_name: str,
):
    cache_key = _compute_hash(texts, model_name)

    base = f"{model_name.replace('/', '_')}_{cache_key}"

    output_path = os.path.join(embed_cfg.output_dir, f"{base}_sentences")
    output_counts_path = os.path.join(embed_cfg.output_dir, f"{base}_counts")

    np.savez_compressed(output_path, embeddings=all_embeddings)
    np.savez_compressed(output_counts_path, counts=np.array(sentence_counts))

    log(f"Saved embeddings → {output_path}", 1)
    log(f"Saved counts     → {output_counts_path}", 1)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    embed_cfg = cfg.embeddings
    device = cfg.run.device
    os.makedirs(embed_cfg.output_dir, exist_ok=True)

    for model_name in embed_cfg.models:
        section(f"MODEL: {model_name}")
        model = SentenceTransformer(model_name, device=device)

        for key, source_cfg in embed_cfg.sources.items():
            section(f"SOURCE: {key}")

            df = load_source(cfg, source_cfg, embed_cfg.input_dir)
            texts = df["text"].dropna().tolist()

            embeddings, counts = compute_embeddings(texts, model, model_name, device, embed_cfg)

            save_embeddings(texts, embeddings, counts, embed_cfg, model_name)

    section("DONE")


if __name__ == "__main__":
    main()