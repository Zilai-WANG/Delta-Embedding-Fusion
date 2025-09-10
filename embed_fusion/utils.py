# utils.py
import os
import json
import time
import random
import numpy as np
import torch
import wandb

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pretty_print_hparams(hp: dict):
    print("Hyperparameters:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

def wandb_run_name(additional_information: str) -> str:
    return f"{additional_information.replace(' ', '_')}_{int(time.time())}"

def init_wandb(project: str, run_name: str, config: dict):
    wandb.init(project=project, name=run_name, config=config)

def save_predictions(predictions, out_path: str):
    """
    predictions: list of (utt_id, predicted_text)
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for utt_id, pred in predictions:
            pred = " ".join(str(pred).strip().split())
            f.write(f"{utt_id} {pred}\n")
    print(f"[SAVE] Wrote predictions to {out_path}")