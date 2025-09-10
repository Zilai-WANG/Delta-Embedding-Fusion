# main.py
import torch
import torch.nn as nn
from transformers import AutoProcessor, AdamW, get_scheduler
import wandb

from utils import (
    load_config, setup_seed, get_device, ensure_dir,
    pretty_print_hparams, init_wandb, wandb_run_name
)
from data import create_train_loader
from model import FusionModelStage2
from evaluation import run_evaluate
from train import train_fusion_model

def main():
    # ===== Load config =====
    cfg = load_config("config.json")
    hp = cfg["hyperparams"]
    paths = cfg["paths"]
    wb = cfg["wandb"]
    model_cfg = cfg["model"]

    # ===== Setup =====
    pretty_print_hparams(hp)
    setup_seed(hp["seed"])
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(hp["save_checkpoint_dir"])

    # ===== Processor / Criterion =====
    processor_wavlm = AutoProcessor.from_pretrained(paths["wavlm_model_path"])
    blank_id = processor_wavlm.tokenizer.pad_token_id
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True).to(device)

    # ===== Model =====
    vocab_size = processor_wavlm.tokenizer.vocab_size
    fusion_model = FusionModelStage2(
        input_dim_wavlm=model_cfg["input_dim_wavlm"],
        input_dim_hubert=model_cfg["input_dim_hubert"],
        hidden_size=hp["hidden_size"],
        vocab_size=vocab_size
    ).to(device)

    print("Number of trainable parameters:",
          sum(p.numel() for p in fusion_model.parameters() if p.requires_grad))

    # ===== DataLoader =====
    train_loader = create_train_loader(
        paths=paths,
        hp=hp,
        processor=processor_wavlm,
        num_workers=hp["num_workers"],
    )

    # ===== Optim + Scheduler =====
    optimizer = AdamW([p for p in fusion_model.parameters() if p.requires_grad],
                      lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

    num_training_steps = max(1, hp["num_of_epochs"] * len(train_loader))
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(hp["warmup_proportion"] * num_training_steps),
        num_training_steps=num_training_steps
    )

    # ===== W&B =====
    init_wandb(
        project=wb["project"],
        run_name=wandb_run_name(hp["additional_information"]),
        config={
            **hp,
            **model_cfg,
            "paths_hash": hash(tuple(paths.values()))  # lightweight run fingerprint
        }
    )
    wandb.watch(fusion_model, log="all", log_freq=100)

    # ===== Eval closure =====
    def eval_closure(return_preds: bool = False):
        return run_evaluate(
            fusion_model=fusion_model,
            embedding_dir_wavlm=paths["eval_embedding_dir_wavlm"],
            embedding_dir_hubert=paths["eval_embedding_dir_hubert"],
            text_file=paths["eval_text_file"],
            processor=processor_wavlm,
            device=device,
            return_preds=return_preds
        )

    # ===== Train =====
    train_fusion_model(
        fusion_model=fusion_model,
        dataloader=train_loader,
        num_epochs=hp["num_of_epochs"],
        save_dir=hp["save_checkpoint_dir"],
        eval_fn=eval_closure,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

if __name__ == "__main__":
    main()