import random, numpy as np, torch

def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_hparams(cfg: dict):
    keys = ["num_of_epochs","learning_rate","weight_decay","batch_size","hidden_size",
            "warmup_proportion","num_workers","dropout","additional_information"]
    print("Hyperparameters:")
    for k in keys:
        if k in cfg:
            print(f"  {k}: {cfg[k]}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    return ap.parse_args()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)