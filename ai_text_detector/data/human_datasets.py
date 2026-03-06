import json
import os
from collections import defaultdict

import hydra
import pandas as pd
from omegaconf import DictConfig


def load_combined_essays(path: str) -> list[dict]:
    essays = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            essays.append(json.loads(line))
    print(f"Loaded {len(essays)} essays to {path}")
    return essays


def save_ivy_panda_essays(essays: list[dict], path: str) -> None:
    rows = []
    for essay in essays:
        title, _, body = essay["text"].partition("\n\n")
        rows.append({"title": title, "text": body})

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {len(rows)} rows → {path}")


def save_asap2_essays(essays: list[dict], path: str) -> None:
    rows = []
    for essay in essays:
        text = essay.get("text")
        extra = essay.get("extra_data")
        if not text or not extra:
            continue
        title = extra.get("prompt_name")
        if not title:
            continue
        rows.append({"title": title, "text": text})

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {len(rows)} rows → {path}")


def save_persuade_essays(essays: list[dict], path: str) -> None:
    essays_by_id: dict[str, list] = defaultdict(list)
    for e in essays:
        essay_id = e["extra_data"]["essay_id_comp"]
        essays_by_id[essay_id].append(e)

    rows = []
    for parts in essays_by_id.values():
        parts_sorted = sorted(parts, key=lambda x: x["extra_data"]["discourse_start"])
        title = parts_sorted[0]["extra_data"].get("prompt_name")
        if title is None:
            continue
        full_text = "\n\n".join(p["text"].strip() for p in parts_sorted)
        rows.append({"title": title, "text": full_text})

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {len(rows)} rows → {path}")


SAVERS = {
    "ivy_panda": save_ivy_panda_essays,
    "asap2": save_asap2_essays,
    "persuade": save_persuade_essays,
}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    os.makedirs(dataset_cfg.output_dir, exist_ok=True)

    essays = load_combined_essays(dataset_cfg.input_path)

    for key, source_cfg in dataset_cfg.sources.items():
        source_name = source_cfg.source_name
        output_path = os.path.join(dataset_cfg.output_dir, source_cfg.output_file)

        subset = [e for e in essays if e["source"] == source_name]
        print(f"Loaded {len(subset)} essays of {source_name}")

        saver = SAVERS.get(key)
        if saver is None:
            print(f"  [!] No saver '{key}', skipping")
            continue

        saver(subset, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
