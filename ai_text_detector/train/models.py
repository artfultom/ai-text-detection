import glob
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import HfApi, upload_file
from huggingface_hub import hf_hub_download, list_repo_files
from sentence_transformers import SentenceTransformer

from ai_text_detector.inference.envs import MODEL_DIR, HF_REPO_ID, HF_TOKEN, DEVICE, ENCODER_NAME

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("models")


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
    log.info(f"Saved model weights → {weights_path}")

    upload_file(
        path_or_fileobj=weights_path,
        path_in_repo=weights_filename,
        repo_id=repo_id,
        token=token,
        commit_message=f"Upload best AE model ({metadata['model']}, roc_auc={metadata['roc_auc']})",
    )
    log.info(f"Uploaded {weights_filename} - hf.co/{repo_id}")

    if readme_template_path and os.path.exists(readme_template_path):
        with open(readme_template_path, "r") as f:
            template = f.read()
        readme_content = template.format_map(metadata)
        log.info(f"Using README template from {readme_template_path}")
    else:
        if readme_template_path:
            log.info(f"WARNING: template not found at {readme_template_path}, using default")
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
    log.info("Uploaded README.md")


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

    log.info(f"Searching HuggingFace repo: {HF_REPO_ID}")
    repo_files = list(list_repo_files(HF_REPO_ID, token=HF_TOKEN))
    pt_files = [f for f in repo_files if f.startswith("ae_") and f.endswith(".pt")]

    if not pt_files:
        raise RuntimeError(f"No ae_*.pt file found in HuggingFace repo {HF_REPO_ID}")

    filename = pt_files[0]
    log.info(f"Downloading checkpoint: {filename}")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN,
        local_dir=MODEL_DIR,
    )
    log.info(f"Saved to: {local_path}")

    return local_path


def load_model_and_encoder():
    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_path = _find_local_checkpoint()
    if ckpt_path:
        log.info(f"Found local checkpoint: {ckpt_path}")
    else:
        log.info(f"No local checkpoint in {MODEL_DIR} — downloading from HuggingFace...")
        ckpt_path = _download_from_hf()

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    metadata: dict = checkpoint.get("metadata", {})
    state_dict = checkpoint["state_dict"]

    dim = state_dict["enc.0.weight"].shape[1]
    ae = AE(dim).to(DEVICE)
    ae.load_state_dict(state_dict)
    ae.eval()
    log.info(f"AE loaded input_dim={dim} metadata={metadata}")

    encoder_name = ENCODER_NAME or metadata.get("model")
    if not encoder_name:
        raise RuntimeError(
            "Cannot determine encoder name. "
            "Set ENCODER_NAME env var or ensure checkpoint metadata contains 'model'."
        )
    log.info(f"Loading sentence encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name, device=DEVICE)

    return ae, encoder, metadata
