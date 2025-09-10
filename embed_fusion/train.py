# train.py
import os
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import wandb
from utils import save_predictions

def train_fusion_model(
    fusion_model,
    dataloader,
    num_epochs: int,
    save_dir: str,
    eval_fn,                 # callable: (return_preds: bool) -> (wer, preds?)
    criterion,               # nn.CTCLoss
    optimizer,
    scheduler,
    device: torch.device
):
    best_wer = float("inf")
    fusion_model.train()
    scaler = GradScaler()

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "fusion_best.pt")
    best_pred_path = os.path.join(save_dir, "predictions_best.txt")

    for epoch in range(num_epochs):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        start_time = time.time()
        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader):
            wavlm_embeddings = batch["wavlm_embeddings"].to(device, non_blocking=True)   # [B, T*, D]
            hubert_embeddings  = batch["hubert_embeddings"].to(device,  non_blocking=True)   # [B, T*, D]
            targets          = batch["targets"].to(device,          non_blocking=True).long()
            target_lengths   = batch["target_lengths"].to(device,   non_blocking=True).long()
            input_lengths    = batch["input_lengths"].to(device,    non_blocking=True).long()

            # Feasibility guard for CTC
            ok = input_lengths >= target_lengths
            feasible_frac = ok.float().mean().item()
            if not torch.all(ok):
                wavlm_embeddings = wavlm_embeddings[ok]
                hubert_embeddings  = hubert_embeddings[ok]
                targets          = targets[ok]
                target_lengths   = target_lengths[ok]
                input_lengths    = input_lengths[ok]
                if wavlm_embeddings.size(0) == 0:
                    wandb.log({"train/feasible_frac": feasible_frac})
                    continue

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                logits = fusion_model(wavlm_embeddings, hubert_embeddings)           # [B, T*, V]
                log_probs = (F.log_softmax(logits, dim=-1)
                               .to(torch.float32)
                               .transpose(0, 1))  # [T*, B, V]
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % 200 == 0 or (step + 1) == num_batches:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr":   lr_now,
                    "train/feasible_frac": feasible_frac,
                    "epoch_progress": epoch + (step + 1) / num_batches,
                })
                print(f"  Epoch {epoch+1}, Step {step+1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, LR: {lr_now:.2e}, Feasible: {feasible_frac:.3f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Last Loss: {loss.item():.4f}")
        print(f"Time taken for epoch {epoch+1}: {time.time() - start_time:.2f} seconds")

        # Evaluate
        eval_wer, _ = eval_fn(return_preds=False)
        print(f"Evaluation WER after epoch {epoch+1}: {eval_wer:.4f}")
        wandb.log({"eval/wer": eval_wer, "epoch": epoch + 1})

        if eval_wer < best_wer:
            best_wer = eval_wer
            torch.save(fusion_model.state_dict(), best_ckpt_path)
            print(f"New best WER {best_wer:.4f}. Model saved to {best_ckpt_path}")

            # Save predictions for the best
            _, preds = eval_fn(return_preds=True)
            save_predictions(preds, best_pred_path)

    wandb.finish()