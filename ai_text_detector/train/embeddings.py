import numpy as np
import torch
import torch.nn.functional as F


def normalize_vec(v: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(v).float()
    t = F.normalize(t, p=2, dim=0)

    return t.numpy()


def compute_poolings(emb: np.ndarray) -> tuple[np.ndarray, ...]:
    mean_val = np.mean(emb, axis=0)

    diffs = emb[1:] - emb[:-1]
    mean_diff_val = np.mean(diffs, axis=0) if len(diffs) else np.zeros_like(mean_val)
    mean_diff_val = normalize_vec(mean_diff_val)

    return (
        mean_val,
        mean_diff_val,
    )
