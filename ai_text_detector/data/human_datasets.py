import pandas as pd
import json
from collections import defaultdict


def download_datasets():
    essays = load_combined_essays("data/combined_essays.jsonl")

    ivy_panda_essays = [essay for essay in essays if essay['source'] == 'IvyPanda Essays']
    print(f"Загружено {len(ivy_panda_essays)} эссе с IvyPanda Essays")
    save_ivy_panda_essays(ivy_panda_essays, "datasets1/ivy_panda_essays.csv")

    asap2_essays = [essay for essay in essays if essay['source'] == 'ASAP2']
    print(f"Загружено {len(asap2_essays)} эссе с ASAP2")
    save_asap2_essays(asap2_essays, "datasets1/asap2_essays.csv")

    persuade_essays = [essay for essay in essays if essay['source'] == 'PERSUADE']
    print(f"Загружено {len(persuade_essays)} эссе с PERSUADE")
    save_persuade_essays(persuade_essays, "datasets1/persuade_essays.csv")


def load_combined_essays(filename):
    essays = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            essays.append(json.loads(line))

    print(f"Загружено {len(essays)} эссе")

    return essays


def save_ivy_panda_essays(essays, filename):
    ivy_panda_rows = []
    for essay in essays:
        text = essay['text']
        title, _, body = text.partition('\n\n')

        ivy_panda_rows.append({"title": title, "text": body})

    ivy_panda_df = pd.DataFrame(ivy_panda_rows)
    ivy_panda_df.to_csv(filename, index=False, encoding="utf-8")


def save_asap2_essays(essays, filename):
    asap2_rows = []
    for essay in essays:
        text = essay['text']
        if not text:
            continue

        extra_data = essay['extra_data']
        if not extra_data:
            continue

        title = extra_data['prompt_name']
        if not title:
            continue

        asap2_rows.append({"title": title, "text": text})

    asap2_df = pd.DataFrame(asap2_rows)
    asap2_df.to_csv(filename, index=False, encoding="utf-8")


def save_persuade_essays(essays, filename):
    essays_by_id = defaultdict(list)
    for e in essays:
        essay_id = e['extra_data']['essay_id_comp']
        essays_by_id[essay_id].append(e)

    data = []
    for essay_id, parts in essays_by_id.items():
        parts_sorted = sorted(parts, key=lambda x: x['extra_data']['discourse_start'])
        full_text = "\n\n".join(p['text'].strip() for p in parts_sorted)

        title = parts_sorted[0]['extra_data'].get('prompt_name')
        if title is None:
            continue

        data.append({
            "title": title,
            "text": full_text
        })

    persuade_df = pd.DataFrame(data)
    persuade_df.to_csv(filename, index=False)
