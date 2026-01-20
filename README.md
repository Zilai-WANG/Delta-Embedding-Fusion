# Delta-Embedding-Fusion

## Overview
**Delta-Embedding-Fusion** is a lightweight framework to improve ASR representations by:

1. **Precomputing embeddings** from self-supervised speech models (e.g., WavLM, HuBERT).  
2. **Computing delta features** = (fine-tuned model hidden states) − (pretrained model hidden states).  
3. **Fusing multiple embedding streams** (e.g., WavLM + HuBERT delta / Wav2Vec2 delta) with a simple, frame-aligned head for CTC decoding.

The repository is divided into two main components:

- **`embed_precompute/`** — Scripts for extracting hidden states and generating delta features.  
- **`embed_fusion/`** — Fusion model training and evaluation.  

---

## Model checkpoints (Hugging Face)
This project uses the following fine-tuned checkpoints from Hugging Face:

- **WavLM**: `balaji1312/wavlm-large-myst-fullfinetune`
- **HuBERT**: `balaji1312/hubert-large-myst-fullfinetune`
- **Wav2Vec2**: `balaji1312/wav2vec2-large-myst-fullfinetune`

