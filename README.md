# Delta-Embedding-Fusion



## Overview

**Delta-Embedding-Fusion** is a lightweight and modular framework designed to enhance Automatic Speech Recognition (ASR) representations through delta-based embedding fusion.

Two steps are:

1. **Precompute embeddings** from self-supervised speech models (e.g., WavLM, HuBERT, Wav2Vec2).  
2. **Compute delta features** as:

   ```
   delta = hidden_states_finetuned − hidden_states_pretrained
   ```

   These delta embeddings capture task-specific adaptation signals.

3. **Fuse multiple embedding streams** (e.g., WavLM delta + HuBERT delta) using a frame-aligned fusion head for CTC decoding.

The framework separates representation extraction from fusion training, saving the computational complexity.

---

## Repository Structure

The repository is organized into two main components:

- `embed_precompute/`  
  Scripts for extracting hidden states from pretrained and fine-tuned models, and for generating delta embeddings.

- `embed_fusion/`  
  Fusion model training, CTC decoding, and evaluation pipelines.

---

## Data Preparation

Please follow the data preparation and split instructions provided in:

https://github.com/Diamondfan/SPAPL_KidsASR?tab=readme-ov-file

Ensure that the dataset structure matches the expected format before running embedding extraction or fusion training.

---

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n fusion python=3.11
conda activate fusion
```

Then install the required dependencies:

```bash
pip install -r fusion_env.yaml
```

(If using `conda env create`, adjust accordingly.)

---

## Model Checkpoints (Hugging Face)

This project relies on the following fine-tuned checkpoints available on Hugging Face:

- **WavLM**  
  `balaji1312/wavlm-large-myst-fullfinetune`

- **HuBERT**  
  `balaji1312/hubert-large-myst-fullfinetune`

- **Wav2Vec2**  
  `balaji1312/wav2vec2-large-myst-fullfinetune`

---

