from sentence_transformers import SentenceTransformer

from ai_text_detector.train.models import AE


class AppState:
    ae: AE
    encoder: SentenceTransformer
    metadata: dict
    device: str
