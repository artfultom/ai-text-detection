import hashlib
import logging
import os
import random

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, upload_file
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ai_text_detector.utils.logging import log, section

logging.getLogger("httpx").setLevel(logging.WARNING)


def _compute_hash(data: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for text in data:
        h.update(text.encode("utf-8"))
    return h.hexdigest()[:32]


def build_datasets(cfg, embed_cfg):
    seed = cfg.run.seed
    input_dir = embed_cfg.input_dir

    ai_paths = [
        os.path.join(input_dir, f.path)
        for f in embed_cfg.sources["ai_data"]
    ]

    df_ai = pd.concat([pd.read_csv(p) for p in ai_paths], ignore_index=True)
    df_ai = df_ai[["text"]].assign(label=1)

    human_path = os.path.join(input_dir, embed_cfg.sources["human_data"][0].path)
    df_human = pd.read_csv(human_path)[["text"]].assign(label=0)

    df_ai = df_ai.dropna(subset=["text"]).reset_index(drop=True)
    df_human = df_human.dropna(subset=["text"]).reset_index(drop=True)

    df_train, df_human_test = train_test_split(
        df_human,
        test_size=len(df_ai),
        random_state=seed,
    )

    df_test = pd.concat([df_ai, df_human_test], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    log(f"Train size: {len(df_train):,} (human)", 1)
    log(f"Test size:  {len(df_test):,} (AI + human)", 1)

    return df_train, df_test


def load_sentence_embeddings(
        embed_dir: str,
        model_name: str,
        texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = _compute_hash(texts, model_name)
    safe_model = model_name.replace("/", "_")

    emb_path = os.path.join(embed_dir, f"{safe_model}_{cache_key}_sentences.npy")
    counts_path = os.path.join(embed_dir, f"{safe_model}_{cache_key}_counts.npy")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not os.path.exists(counts_path):
        raise FileNotFoundError(f"Counts not found: {counts_path}")

    embeddings = np.load(emb_path)
    counts = np.load(counts_path)

    log(f"Loaded embeddings {embeddings.shape} \t- {os.path.basename(emb_path)}", 2)
    log(f"Loaded counts     {counts.shape}     \t- {os.path.basename(counts_path)}", 2)

    return embeddings, counts


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(v).float()
    t = F.normalize(t, p=2, dim=0)

    return t.numpy()


def compute_poolings(emb: np.ndarray) -> tuple[np.ndarray, ...]:
    mean_val = np.mean(emb, axis=0)

    diffs = emb[1:] - emb[:-1]
    mean_diff_val = np.mean(diffs, axis=0) if len(diffs) else np.zeros_like(mean_val)
    mean_diff_val = _normalize_vec(mean_diff_val)

    return (
        mean_val,
        mean_diff_val,
    )


POOLING_NAMES = [
    "mean",
    "mean_diff",
]


def counts_to_spans(counts: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for n in counts:
        end = start + int(n)
        spans.append((start, end))
        start = end
    return spans


def build_document_poolings(
        embeddings: np.ndarray,
        counts: np.ndarray,
) -> list[list[np.ndarray]]:
    spans = counts_to_spans(counts)
    pooling_lists: list[list[np.ndarray]] = [[] for _ in range(len(POOLING_NAMES))]

    for start, end in tqdm(spans, desc="Pooling docs", leave=False):
        vecs = compute_poolings(embeddings[start:end])
        for i, v in enumerate(vecs):
            pooling_lists[i].append(v)

    return pooling_lists


class AE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(
        ae: AE,
        X_train: np.ndarray,
        device: str,
        n_epochs: int,
        batch_size: int,
        lr: float,
        val_fraction: float = 0.1,
        patience: int = 5,
) -> None:
    n_val = max(1, int(len(X_train) * val_fraction))
    X_val = X_train[:n_val]
    X_tr = X_train[n_val:]

    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val = float("inf")
    no_improve = 0
    best_state = None

    for _ in range(n_epochs):
        ae.train()
        for (batch,) in loader:
            batch = batch.to(device)
            loss = loss_fn(ae(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        ae.eval()
        with torch.no_grad():
            val_loss = loss_fn(ae(X_val_t), X_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        ae.load_state_dict(best_state)
        ae.to(device)


def upload_best_model_to_hf(
        ae: AE,
        metadata: dict,
        repo_id: str,
        token: str,
        output_dir: str,
        readme_template_path: str | None = None,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    safe_model = metadata["model"].replace("/", "_")
    weights_filename = f"ae_{safe_model}_best.pt"
    weights_path = os.path.join(output_dir, weights_filename)
    torch.save({"state_dict": ae.state_dict(), "metadata": metadata}, weights_path)
    log(f"Saved model weights → {weights_path}", 1)

    upload_file(
        path_or_fileobj=weights_path,
        path_in_repo=weights_filename,
        repo_id=repo_id,
        token=token,
        commit_message=f"Upload best AE model ({metadata['model']}, roc_auc={metadata['roc_auc']})",
    )
    log(f"Uploaded {weights_filename} - hf.co/{repo_id}", 1)

    if readme_template_path and os.path.exists(readme_template_path):
        with open(readme_template_path, "r") as f:
            template = f.read()
        readme_content = template.format_map(metadata)
        log(f"Using README template from {readme_template_path}", 1)
    else:
        if readme_template_path:
            log(f"WARNING: template not found at {readme_template_path}, using default", 1)
        readme_content = f"# AI Text Detector\n\nModel: `{metadata['model']}`\nROC-AUC: {metadata['roc_auc']}\n"

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="Add README",
    )
    log("Uploaded README.md", 1)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_cfg = cfg.train
    embed_cfg = cfg.embeddings
    device = cfg.run.device
    seed = cfg.run.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(train_cfg.output_dir, exist_ok=True)

    all_rows = []
    best_ae: AE | None = None
    best_roc_auc: float = -1.0
    best_metadata: dict = {}

    for model_name in train_cfg.models:
        section(f"MODEL: {model_name}")

        section("BUILD DATASETS")
        df_train, df_test = build_datasets(cfg, embed_cfg)

        train_texts = df_train["text"].tolist()
        test_texts = df_test["text"].tolist()
        y_test = df_test["label"].values.astype(int)

        log("Loading train embeddings...", 1)
        train_emb_raw, train_counts = load_sentence_embeddings(
            embed_cfg.output_dir, model_name, train_texts
        )

        log("Loading test embeddings...", 1)
        test_emb_raw, test_counts = load_sentence_embeddings(
            embed_cfg.output_dir, model_name, test_texts
        )

        log("Building poolings...", 1)
        train_pools = build_document_poolings(train_emb_raw, train_counts)
        test_pools = build_document_poolings(test_emb_raw, test_counts)

        mean_train = np.array(train_pools[0])
        mean_test = np.array(test_pools[0])

        X_train = np.concatenate([mean_train * 1.0, np.array(train_pools[1]) * 1.2], axis=1)
        X_test = np.concatenate([mean_test * 1.0, np.array(test_pools[1]) * 1.2], axis=1)

        log("Training AE...", 1)
        ae = AE(X_train.shape[1]).to(device)
        train_ae(
            ae,
            X_train,
            device=device,
            n_epochs=train_cfg.ae_epochs,
            batch_size=train_cfg.ae_batch_size,
            lr=train_cfg.ae_lr,
            val_fraction=train_cfg.ae_val_fraction,
            patience=train_cfg.ae_patience,
        )

        ae.eval()
        with torch.no_grad():
            recon = ae(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        scores = np.mean((X_test - recon) ** 2, axis=1)

        roc_auc = round(roc_auc_score(y_test, scores), 4)
        pr_auc = round(average_precision_score(y_test, scores), 4)

        row = {
            "model": model_name,
            "pooling": "mean + mean_diff",
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        all_rows.append(row)
        log(f"  pooling='mean + mean_diff' — ROC-AUC={roc_auc}, PR-AUC={pr_auc}", 1)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_ae = ae
            best_metadata = row

    section("RESULTS")

    df_results = pd.DataFrame(all_rows)
    print(df_results[["model", "pooling", "roc_auc", "pr_auc"]].to_string(index=False))

    out_path = os.path.join(train_cfg.output_dir, "results.csv")
    df_results.to_csv(out_path, index=False)
    log(f"Full results saved - {out_path}", 1)

    hf_cfg = train_cfg.get("huggingface", None)
    if hf_cfg and hf_cfg.get("upload", False) and best_ae is not None:
        section("UPLOAD TO HUGGING FACE")
        upload_best_model_to_hf(
            ae=best_ae,
            metadata=best_metadata,
            repo_id=hf_cfg.repo_id,
            token=hf_cfg.get("token") or os.environ["HF_TOKEN"],
            output_dir=train_cfg.output_dir,
            readme_template_path=hf_cfg.get("readme_template"),
        )

    section("DONE")


if __name__ == "__main__":
    main()
