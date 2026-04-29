FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.serve.txt .
RUN pip install --no-cache-dir -r requirements.serve.txt

COPY serve.py .
COPY ai_text_detector ./ai_text_detector

RUN mkdir -p /app/train_results
COPY train_results ./train_results

ENV MODEL_DIR=/app/train_results \
    DEVICE=cpu \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

RUN pip install sentence-transformers \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"


CMD ["python", "-m", "uvicorn", "serve:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--log-level", "info"]