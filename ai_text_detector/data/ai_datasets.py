import pandas as pd
import math
from llama_cpp import Llama
import re
import os

EXAMPLE_ID = 0
START = 1000
OUT_COUNT = 1

SEED = 42


PROMPTS = [
    ("""Write an essay of about {} words on the topic '{}'.
    Start directly with the first sentence of the essay. Absolutely no title or heading of any kind. 
    Output only the essay text. """),

    ("""Write an essay of about {} words on the topic '{}', following the stylistic and structural conventions of IvyPanda essays.
    
    STYLE REQUIREMENTS (IvyPanda-like):
    1. The tone must be academic, clear, neutral, and objective.
    2. The essay should include background context, a clear analytical structure, and well-developed paragraphs.
    3. Arguments must be supported with reasoning, concise explanations, and relevant examples.
    4. The writing must be coherent, logically progressive, and easy to read.
    5. Avoid emotional language or overly dramatic phrasing.
    
    STRUCTURE REQUIREMENTS:
    1. Divide the essay into several sections, each starting with a plain-text subheading.
    2. Subheadings must contain only letters and spaces — no punctuation, no symbols, and absolutely no Markdown formatting.
       - This means: no asterisks, no bold, no italics, no underscores, no brackets.
       - Subheadings must appear as plain text lines, e.g. Background Context
    3. Each subheading must be on its own line with exactly one blank line before and after it.
    4. No overall essay title — begin directly with the first subheading.
    5. Output must be plain text only (no Markdown, no asterisks, no bold, no italics, no hashtags, no special formatting).
    
    OPTIONAL ELEMENTS:
    You may include a plain-text references section at the end if appropriate to the topic, but this is optional.
    
    Begin with the first subheading, then continue with the first paragraph.
    Output only the essay text."""),

    ("""You are a helpful writing assistant. Your task is to generate essays of a given length and topic.
    Start directly with the first sentence. No titles or headings. Output only the essay text.
    
    Here is an example:
    
    Example topic: "{}"
    Example length: {} words
    
    Example output:
    {}
    
    Now generate a new essay.
    
    Topic: "{}"
    Length: {} words
    
    Write the essay now."""),

    ("""Identify the writing style and format of the text, excluding all semantic content and themes.
    
    Text:
    {}""",
     """Write an essay of about {} words on the topic '{}'.
     
     You are given the following style-and-format description:
     '{}'
     
     Start directly with the first sentence of the essay. Absolutely no title or heading of any kind.
     Output only the essay text."""),

    ("""You are an academic writing instructor.
    Create a well-structured outline for an academic essay on the following topic: {}.
    
    Requirements:
    
    The outline must include the following sections:
    - Introduction
    - Main Body (2–4 thematic sections)
    - Counterarguments / Alternative Perspectives
    - Conclusion
    
    Each section should contain 2–3 concise bullet points describing the key ideas or arguments.
    The outline should reflect a clear thesis-driven structure.
    Use formal academic language.
    
    Do not write the essay itself""",
     """You are an academic essay writer.
     Write a coherent academic essay of about {} words on the topic '{}' strictly following the outline below.
     
     Outline:
     {}
     
     Requirements:
     - Follow the outline in the given order without adding or removing sections.
     - Develop each bullet point into a full, well-argued paragraph.
     - Maintain a formal academic tone and clear argumentative logic.
     - Use appropriate academic transitions and signposting.
     - Do not introduce new arguments or themes not present in the outline.
     - Avoid informal language, personal anecdotes, and unsupported claims."""),
]


def load_model(model_path, n_ctx, n_gpu_layers=-1):
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=1024,
        n_threads=12,
        use_mlock=True,
        use_mmap=True,
        metal=True,
        verbose=False,
        seed=SEED
    )

    return llm


def generate_response(llm, prompt, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1):
    output = llm(
        prompt,
        max_tokens=llm.n_ctx() // 2,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )

    return output['choices'][0]['text']


def word_count(text):
    words = re.findall(r'\b\w+\b', text)
    return math.floor(len(words) / 100) * 100


def generate(model, model_name, df, prompt_raw, start=0, n=5000):
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])

        prompt_formatted = prompt_raw.format(length, title)

        try:
            response = generate_response(
                model,
                prompt_formatted,
                temperature=0.7,
            )
        except:
            continue

        response = response.lstrip()

        rows.append({"title": title, "prompt": prompt_formatted, "text": response, "model": model_name})

    return pd.DataFrame(rows)


def generate_few_shot(model, model_name, df, prompt_raw, start=0, n=5000):
    rows = []
    for i in range(start + 1, start + n + 1):
        prev_title = df.loc[i - 1, "title"]
        prev_length = word_count(df.loc[i - 1, "text"])
        prev_text = df.loc[i - 1, "text"]

        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])

        prompt_formatted = prompt_raw.format(prev_title, prev_length, prev_text, title, length)

        try:
            response = generate_response(
                model,
                prompt_formatted,
                temperature=0.7,
            )
        except:
            continue

        response = response.lstrip()

        rows.append({"title": title, "prompt": prompt_formatted, "text": response, "model": model_name})

    return pd.DataFrame(rows)


def generate_with_content(model, model_name, df, prompt_raw_1, prompt_raw_2, start=0, n=5000):
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])
        text = df.loc[i, "text"]

        prompt_1_formatted = prompt_raw_1.format(text)

        try:
            response_1 = generate_response(
                model,
                prompt_1_formatted,
                temperature=0.7,
            )
            response_1 = response_1.lstrip()

            prompt_2_formatted = prompt_raw_2.format(length, title, response_1)
            response_2 = generate_response(
                model,
                prompt_2_formatted,
                temperature=0.7,
            )
        except:
            continue

        response = response_2.lstrip()

        rows.append({
            "title": title,
            "prompt_1": prompt_1_formatted,
            "response": response_1,
            "prompt_2": prompt_2_formatted,
            "text": response,
            "model": model_name
        })

    return pd.DataFrame(rows)


def generate_with_plan(model, model_name, df, prompt_raw_1, prompt_raw_2, start=0, n=5000):
    rows = []
    for i in range(start, start + n):
        title = df.loc[i, "title"]
        length = word_count(df.loc[i, "text"])

        prompt_1_formatted = prompt_raw_1.format(title)

        try:
            response_1 = generate_response(
                model,
                prompt_1_formatted,
                temperature=0.7,
            )
            response_1 = response_1.lstrip()

            prompt_2_formatted = prompt_raw_2.format(length, title, response_1)
            response_2 = generate_response(
                model,
                prompt_2_formatted,
                temperature=0.7,
            )
        except:
            continue

        response = response_2.lstrip()

        rows.append({
            "title": title,
            "prompt_1": prompt_1_formatted,
            "response": response_1,
            "prompt_2": prompt_2_formatted,
            "text": response,
            "model": model_name
        })

    return pd.DataFrame(rows)


def save_dataset(out_file_prefix, index, df):
    file_path = f"datasets1/{out_file_prefix}_{index}.csv"
    df.to_csv(
        file_path,
        mode="a",
        header=not os.path.exists(file_path),
        index=False,
        encoding="utf-8",
    )


def generate_datasets():
    df = pd.read_csv("datasets/ivy_panda_essays.csv", encoding="utf-8")

    generate_local_llm_dataset(
        'models/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
        'mistral-7b-instruct-v0.2.Q5_K_M',
        32768,
        'mistral_essays',
        "[INST]", "[/INST]",
        df
    )

    generate_local_llm_dataset(
        'models/Llama-3-13B-Instruct-v0.1.Q5_K_M.gguf',
        'llama-3-13B-Instruct-v0.1.Q5_K_M',
        8192,
        'llama_essays',
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        df
    )


def generate_local_llm_dataset(model_path, model_name, n_ctx, out_file_prefix, prompt_start, prompt_end, df):
    llm = load_model(model_path, n_ctx=n_ctx)

    df1 = generate(llm, model_name, df, f"{prompt_start}\n{PROMPTS[0]}\n{prompt_end}", n=OUT_COUNT)
    save_dataset(out_file_prefix, 1, df1)

    df2 = generate(llm, model_name, df, f"{prompt_start}\n{PROMPTS[1]}\n{prompt_end}", n=OUT_COUNT)
    save_dataset(out_file_prefix, 2, df2)

    df3 = generate_few_shot(llm, model_name, df, f"{prompt_start}\n{PROMPTS[2]}\n{prompt_end}", n=OUT_COUNT)
    save_dataset(out_file_prefix, 3, df3)

    df4 = generate_with_content(
        llm,
        model_name,
        df,
        f"{prompt_start}\n{PROMPTS[3][0]}\n{prompt_end}",
        f"{prompt_start}\n{PROMPTS[3][1]}\n{prompt_end}",
        n=OUT_COUNT
    )
    save_dataset(out_file_prefix, 4, df4)

    df5 = generate_with_plan(
        llm,
        model_name,
        df,
        f"{prompt_start}\n{PROMPTS[4][0]}\n{prompt_end}",
        f"{prompt_start}\n{PROMPTS[4][1]}\n{prompt_end}",
        n=OUT_COUNT
    )
    save_dataset(out_file_prefix, 5, df5)
