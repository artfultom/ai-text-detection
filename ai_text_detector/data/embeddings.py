import hashlib
import os
import re

import hydra
import nltk
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_essays(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded {len(df)} essays from {path}")
    return df


def _compute_hash(data, model_name):
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))

    for text in data:
        h.update(text.encode("utf-8"))

    return h.hexdigest()[:32]


def compute_embeddings(
        data: list[str],
        model: SentenceTransformer,
        model_name: str,
        device: str,
        embed_cfg,
        norm: bool,
) -> tuple[np.ndarray, list[int]]:
    print(f"Computing embeddings for {model_name}...")

    all_sentences = []
    sentence_counts = []
    for text in tqdm(data):
        text = re.sub(r"\n+", ". ", text)
        sentences = nltk.sent_tokenize(text)

        if "e5" in model_name:
            sentences = [f"passage: {s}" for s in sentences]

        all_sentences.extend(sentences)
        sentence_counts.append(len(sentences))

    all_embeddings = model.encode(
        all_sentences,
        batch_size=embed_cfg.batch_size,
        normalize_embeddings=norm,
        show_progress_bar=True,
        device=device,
    )

    return all_embeddings, sentence_counts


def save_embeddings(
        texts,
        all_embeddings: np.ndarray,
        sentence_counts: list[int],
        embed_cfg,
        key: str,
        model_name: str,
        norm: bool,
) -> None:
    norm_str = "_norm" if norm else ""
    cache_key = _compute_hash(texts, model_name)

    output_path = os.path.join(
        embed_cfg.output_dir,
        f"{key}_{model_name.replace('/', '_')}{norm_str}_{cache_key}_sentences",
    )
    output_counts_path = os.path.join(
        embed_cfg.output_dir,
        f"{key}_{model_name.replace('/', '_')}{norm_str}_{cache_key}_counts",
    )

    np.savez_compressed(output_path, embeddings=all_embeddings)
    np.savez_compressed(output_counts_path, counts=np.array(sentence_counts))

    print(f"{len(sentence_counts)} essays:")
    print(f"  Saved embeddings → {output_path}")
    print(f"  Saved counts → {output_counts_path}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    embed_cfg = cfg.embeddings
    device = cfg.run.device
    os.makedirs(embed_cfg.output_dir, exist_ok=True)

    for model_name in embed_cfg.models:
        print(f"\n=== Model: {model_name} ===")
        model = SentenceTransformer(model_name)

        for key, source_cfg in embed_cfg.sources.items():
            print(f"\n  Source: {key}")
            for norm in [True, False]:
                input_path = os.path.join(embed_cfg.input_dir, source_cfg.input_file)

                df = load_essays(input_path)
                texts = df["text"].dropna().tolist()

                all_embeddings, sentence_counts = compute_embeddings(
                    texts, model, model_name, device, embed_cfg, norm
                )
                save_embeddings(texts, all_embeddings, sentence_counts, embed_cfg, key, model_name, norm)

    print("Done.")


if __name__ == "__main__":
    main()
