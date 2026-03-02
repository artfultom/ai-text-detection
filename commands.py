import fire

from ai_text_detector.data.human_datasets import download_datasets
from ai_text_detector.data.ai_datasets import generate_datasets

if __name__ == "__main__":
    fire.Fire(
        {
            "download_datasets": download_datasets,
            "generate_ai": generate_datasets,
        }
    )
