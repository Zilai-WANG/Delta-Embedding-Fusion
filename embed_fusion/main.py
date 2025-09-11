import argparse, time, os
import torch
import torch.nn as nn
from transformers import AutoProcessor, AdamW, get_scheduler
import wandb, yaml

from utils import setup_seed, print_hparams, parse_args, load_cfg
from model import FusionModelStage2
from data import FusionEmbeddingDataset, collate_fn
from evaluation import run_evaluate
from train import train_fusion_model



def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # ---- Seed & Device ----
    setup_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    os.makedirs(cfg["save_checkpoint_dir"], exist_ok=True)

    # ---- Log hyperparams ----
    print_hparams(cfg)

    # ---- Processor / Loss ----
    processor_wavlm = AutoProcessor.from_pretrained(cfg["wavlm_model_path"])
    blank_id = processor_wavlm.tokenizer.pad_token_id
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True).to(device)

    # ---- Model ----
    input_dim_wavlm = 1024
    input_dim_hubert = 1024
    vocab_size = processor_wavlm.tokenizer.vocab_size
    fusion_model = FusionModelStage2(
        input_dim_wavlm, input_dim_hubert, cfg["hidden_size"], vocab_size, dropout=cfg["dropout"]
    ).to(device)
    print("Number of trainable parameters:",
          sum(p.numel() for p in fusion_model.parameters() if p.requires_grad))

    # ---- Data ----
    train_dataset = FusionEmbeddingDataset(
        cfg["train_embedding_dir_wavlm"],
        cfg["train_embedding_dir_w2v2"],
        cfg["train_text_file"],
        processor_wavlm
    )
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    # ---- Optim + Sched ----
    optimizer = AdamW([p for p in fusion_model.parameters() if p.requires_grad],
                      lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    num_training_steps = max(1, cfg["num_of_epochs"] * len(train_loader))
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(cfg["warmup_proportion"] * num_training_steps),
        num_training_steps=num_training_steps
    )

    # ---- W&B ----
    wandb.init(
        project=cfg["wandb_project"],
        name=f"{cfg['additional_information'].replace(' ', '_')}_{int(time.time())}",
        config=cfg
    )
    wandb.watch(fusion_model, log="all", log_freq=100)

    # ---- Eval closure ----
    def eval_closure():
        return run_evaluate(
            fusion_model,
            cfg["eval_embedding_dir_wavlm"],
            cfg["eval_embedding_dir_w2v2"],
            cfg["eval_text_file"],
            processor_wavlm,
            device
        )

    # ---- Train ----
    train_fusion_model(
        fusion_model=fusion_model,
        dataloader=train_loader,
        num_epochs=cfg["num_of_epochs"],
        save_dir=cfg["save_checkpoint_dir"],
        eval_fn=eval_closure,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

if __name__ == "__main__":
    main()