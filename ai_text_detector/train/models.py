import os

import torch
import torch.nn as nn
from huggingface_hub import HfApi, upload_file

from ai_text_detector.utils.logging import log


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
