import math
import os
import re

import hydra
import pandas as pd
from llama_cpp import Llama
from omegaconf import DictConfig


def load_model(cfg: DictConfig) -> Llama:
    return Llama(
        model_path=cfg.model.path,
        n_ctx=cfg.model.n_ctx,
        n_gpu_layers=cfg.model.n_gpu_layers,
        n_batch=cfg.model.n_batch,
        n_threads=cfg.model.n_threads,
        use_mlock=True,
        use_mmap=True,
        metal=True,
        verbose=False,
        seed=cfg.run.seed,
    )


def generate_response(llm: Llama, prompt: str, cfg: DictConfig) -> str:
    output = llm(
        prompt,
        max_tokens=llm.n_ctx() // 2,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        repeat_penalty=cfg.generation.repeat_penalty,
    )
    return output["choices"][0]["text"]


def word_count(text: str) -> int:
    words = re.findall(r"\b\w+\b", text)
    return math.floor(len(words) / 100) * 100


def wrap_prompt(raw: str, cfg: DictConfig) -> str:
    return f"{cfg.model.prompt_start}\n{raw}{cfg.model.prompt_end}"


def save_dataset(cfg: DictConfig, index: int, df: pd.DataFrame) -> None:
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    file_path = os.path.join(cfg.dataset.output_dir, f"{cfg.model.out_file_prefix}_{index}.csv")
    df.to_csv(
        file_path,
        mode="a",
        header=not os.path.exists(file_path),
        index=False,
        encoding="utf-8",
    )
    print(f"  Saved {len(df)} rows → {file_path}")


def generate(llm, model_name, df, prompt_raw, cfg):
    start, n = cfg.run.start, cfg.run.count
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])
        prompt = prompt_raw.format(length, title)
        try:
            response = generate_response(llm, prompt, cfg).lstrip()
        except Exception as e:
            print(f"  [strategy 1] row {i} failed: {e}")
            continue
        rows.append({"title": title, "prompt": prompt, "text": response, "model": model_name})
    return pd.DataFrame(rows)


def generate_few_shot(llm, model_name, df, prompt_raw, cfg):
    start, n = cfg.run.start, cfg.run.count
    rows = []
    for i in range(start + 1, start + n + 1):
        prev_title = df.loc[i - 1, "title"]
        prev_length = word_count(df.loc[i - 1, "text"])
        prev_text = df.loc[i - 1, "text"]
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])
        prompt = prompt_raw.format(prev_title, prev_length, prev_text, title, length)
        try:
            response = generate_response(llm, prompt, cfg).lstrip()
        except Exception as e:
            print(f"  [strategy 3] row {i} failed: {e}")
            continue
        rows.append({"title": title, "prompt": prompt, "text": response, "model": model_name})
    return pd.DataFrame(rows)


def generate_with_content(llm, model_name, df, prompt_raw_1, prompt_raw_2, cfg):
    start, n = cfg.run.start, cfg.run.count
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])
        text = df.loc[i, "text"]
        prompt_1 = prompt_raw_1.format(text)
        try:
            response_1 = generate_response(llm, prompt_1, cfg).lstrip()
            prompt_2 = prompt_raw_2.format(length, title, response_1)
            response_2 = generate_response(llm, prompt_2, cfg).lstrip()
        except Exception as e:
            print(f"  [strategy 4] row {i} failed: {e}")
            continue
        rows.append({
            "title": title,
            "prompt_1": prompt_1,
            "response": response_1,
            "prompt_2": prompt_2,
            "text": response_2,
            "model": model_name,
        })
    return pd.DataFrame(rows)


def generate_with_plan(llm, model_name, df, prompt_raw_1, prompt_raw_2, cfg):
    start, n = cfg.run.start, cfg.run.count
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])
        prompt_1 = prompt_raw_1.format(title)
        try:
            response_1 = generate_response(llm, prompt_1, cfg).lstrip()
            prompt_2 = prompt_raw_2.format(length, title, response_1)
            response_2 = generate_response(llm, prompt_2, cfg).lstrip()
        except Exception as e:
            print(f"  [strategy 5] row {i} failed: {e}")
            continue
        rows.append({
            "title": title,
            "prompt_1": prompt_1,
            "response": response_1,
            "prompt_2": prompt_2,
            "text": response_2,
            "model": model_name,
        })
    return pd.DataFrame(rows)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Loading dataset from {cfg.dataset.human_path} ...")
    df = pd.read_csv(cfg.dataset.human_path, encoding="utf-8")

    print(f"Loading model {cfg.model.name} ...")
    llm = load_model(cfg)
    model_name = cfg.model.name
    strategies = list(cfg.generation.strategies)

    print(f"Running strategies {strategies}, start={cfg.run.start}, count={cfg.run.count}")

    p = cfg.prompts

    if 1 in strategies:
        print("Strategy 1: zero-shot basic")
        df1 = generate(llm, model_name, df, wrap_prompt(p.strategy_1.prompt, cfg), cfg)
        save_dataset(cfg, 1, df1)

    if 2 in strategies:
        print("Strategy 2: IvyPanda style")
        df2 = generate(llm, model_name, df, wrap_prompt(p.strategy_2.prompt, cfg), cfg)
        save_dataset(cfg, 2, df2)

    if 3 in strategies:
        print("Strategy 3: few-shot")
        df3 = generate_few_shot(llm, model_name, df, wrap_prompt(p.strategy_3.prompt, cfg), cfg)
        save_dataset(cfg, 3, df3)

    if 4 in strategies:
        print("Strategy 4: style extraction + generation")
        df4 = generate_with_content(
            llm, model_name, df,
            wrap_prompt(p.strategy_4.prompt_1, cfg),
            wrap_prompt(p.strategy_4.prompt_2, cfg),
            cfg,
        )
        save_dataset(cfg, 4, df4)

    if 5 in strategies:
        print("Strategy 5: outline + essay")
        df5 = generate_with_plan(
            llm, model_name, df,
            wrap_prompt(p.strategy_5.prompt_1, cfg),
            wrap_prompt(p.strategy_5.prompt_2, cfg),
            cfg,
        )
        save_dataset(cfg, 5, df5)

    print("Done.")


if __name__ == "__main__":
    main()
