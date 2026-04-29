---
language:
  - en
tags:
  - ai-text-detection
  - autoencoder
  - anomaly-detection
  - text-classification
license: mit
---

# 🔍 AI Text Detector

> Unsupervised autoencoder trained on human-written text.  
> Detects AI-generated content by measuring reconstruction error.

## How it works

The model is trained **only on human text**. At inference time, AI-generated
text produces higher reconstruction error - this score is used as the
detection signal.

Sentence embeddings are computed with `{model}`, then pooled into two
document-level features:

| Feature             | Description |
|---------------------|---|
| `mean pooling`      | Average sentence embedding |
| `mean diff pooling` | Normalised mean of consecutive embedding deltas |

## Performance

| Metric | Value |
|---|---|
| ROC-AUC | **{roc_auc}** |
| PR-AUC | **{pr_auc}** |

## Architecture

| Parameter    | Value |
|--------------|---|
| Backbone     | `{model}` |
| Pooling      | `{pooling}` |
