import hashlib
import logging
import os
import random

import hydra
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ai_text_detector.train.embeddings import compute_poolings
from ai_text_detector.train.models import AE, upload_best_model_to_hf

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train")

optuna.logging.set_verbosity(optuna.logging.WARNING)


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

    log.info(f"Train size: {len(df_train):,} (human)")
    log.info(f"Test size:  {len(df_test):,} (AI + human)")

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

    log.info(f"Loaded embeddings {embeddings.shape} \t- {os.path.basename(emb_path)}")
    log.info(f"Loaded counts     {counts.shape}     \t- {os.path.basename(counts_path)}")

    return embeddings, counts


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


def _build_feature_matrix(
        pools: list[list[np.ndarray]],
) -> np.ndarray:
    return np.concatenate([np.array(pools[0]), np.array(pools[1])], axis=1)


def _train_ae_with_params(
        X_train: np.ndarray,
        lr: float,
        n_epochs: int,
        batch_size: int,
        device: str,
) -> AE:
    d = X_train.shape[1]
    ae = AE(d).to(device)
    ae.train()

    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for _ in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            loss = loss_fn(ae(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return ae


def _reconstruction_scores(ae: AE, X: np.ndarray, device: str) -> np.ndarray:
    ae.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = ae(Xt).cpu().numpy()

    return np.mean((X - recon) ** 2, axis=1)


def objective(
        trial: optuna.Trial,
        train_pools: list[list[np.ndarray]],
        test_pools: list[list[np.ndarray]],
        y_test: np.ndarray,
        train_cfg,
        device: str,
) -> float:
    lr = trial.suggest_float("lr", train_cfg.lr_min, train_cfg.lr_max, log=True)
    n_epochs = trial.suggest_int("n_epochs", train_cfg.ae_epochs_min, train_cfg.ae_epochs_max)
    batch_size = trial.suggest_categorical("batch_size", train_cfg.batch_sizes)

    X_train = _build_feature_matrix(train_pools)
    X_test = _build_feature_matrix(test_pools)

    ae = _train_ae_with_params(
        X_train=X_train,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
    )

    scores = _reconstruction_scores(ae, X_test, device)
    return roc_auc_score(y_test, scores)


def run_optuna_study(
        train_pools: list[list[np.ndarray]],
        test_pools: list[list[np.ndarray]],
        y_test: np.ndarray,
        device: str,
        n_trials: int,
        train_cfg,
        seed: int,
) -> tuple[dict, pd.DataFrame]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, train_pools, test_pools, y_test, train_cfg, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    log.info(f"  Best trial: #{study.best_trial.number} — ROC-AUC={study.best_value:.4f}")
    log.info(f"  Best params: {study.best_params}")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)

    return study.best_params, trials_df


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

    n_trials = int(getattr(train_cfg, "optuna_n_trials", 30))

    all_rows = []
    all_optuna_rows = []
    best_ae: AE | None = None
    best_roc_auc: float = -1.0
    best_metadata: dict = {}

    for model_name in train_cfg.models:
        log.info(f"MODEL: {model_name}")

        log.info("BUILD DATASETS")
        df_train, df_test = build_datasets(cfg, embed_cfg)

        train_texts = df_train["text"].tolist()
        test_texts = df_test["text"].tolist()
        y_test = df_test["label"].values.astype(int)

        log.info("Loading train embeddings...")
        train_emb_raw, train_counts = load_sentence_embeddings(
            embed_cfg.output_dir, model_name, train_texts
        )

        log.info("Loading test embeddings...")
        test_emb_raw, test_counts = load_sentence_embeddings(
            embed_cfg.output_dir, model_name, test_texts
        )

        log.info("Building poolings...")
        train_pools = build_document_poolings(train_emb_raw, train_counts)
        test_pools = build_document_poolings(test_emb_raw, test_counts)

        log.info(f"Running Optuna ({n_trials} trials)...")
        best_params, trials_df = run_optuna_study(
            train_pools=train_pools,
            test_pools=test_pools,
            y_test=y_test,
            device=device,
            n_trials=n_trials,
            train_cfg=train_cfg,
            seed=seed,
        )

        trials_df.insert(0, "model", model_name)
        all_optuna_rows.append(trials_df)

        log.info("Training final AE with best hyperparameters...")
        X_train_final = _build_feature_matrix(train_pools)
        X_test_final = _build_feature_matrix(test_pools)

        ae = _train_ae_with_params(
            X_train=X_train_final,
            lr=best_params["lr"],
            n_epochs=best_params["n_epochs"],
            batch_size=best_params["batch_size"],
            device=device,
        )

        scores = _reconstruction_scores(ae, X_test_final, device)
        roc_auc = round(roc_auc_score(y_test, scores), 4)
        pr_auc = round(average_precision_score(y_test, scores), 4)

        fpr, tpr, thresholds = roc_curve(y_test, scores)
        optimal_idx = (tpr - fpr).argmax()
        optimal_threshold = thresholds[optimal_idx]

        row = {
            "model": model_name,
            "pooling": "mean + mean_diff",
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": optimal_threshold,
            **best_params,
        }
        all_rows.append(row)
        log.info(f"  Final — ROC-AUC={roc_auc}, PR-AUC={pr_auc}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_ae = ae
            best_metadata = row

    log.info("RESULTS")

    df_results = pd.DataFrame(all_rows)
    print(df_results[["model", "pooling", "roc_auc", "pr_auc", "threshold"]].to_string(index=False))

    out_path = os.path.join(train_cfg.output_dir, "results.csv")
    df_results.to_csv(out_path, index=False)
    log.info(f"Full results saved - {out_path}")

    if all_optuna_rows:
        optuna_path = os.path.join(train_cfg.output_dir, "optuna_results.csv")
        pd.concat(all_optuna_rows, ignore_index=True).to_csv(optuna_path, index=False)
        log.info(f"Optuna trials saved - {optuna_path}")

    hf_cfg = train_cfg.get("huggingface", None)
    if hf_cfg and hf_cfg.get("upload", False) and best_ae is not None:
        log.info("UPLOAD TO HUGGING FACE")
        upload_best_model_to_hf(
            ae=best_ae,
            metadata=best_metadata,
            repo_id=hf_cfg.repo_id,
            token=hf_cfg.get("token") or os.environ["HF_TOKEN"],
            output_dir=train_cfg.output_dir,
            readme_template_path=hf_cfg.get("readme_template"),
        )

    log.info("DONE")


if __name__ == "__main__":
    main()
