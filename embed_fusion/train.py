import os, time, torch, torch.nn.functional as F, wandb
from torch.amp import autocast, GradScaler

def train_fusion_model(fusion_model, dataloader, num_epochs, save_dir,
                       eval_fn, criterion, optimizer, scheduler, device):
    best_wer = float("inf")
    scaler = GradScaler()
    num_batches = len(dataloader)

    for epoch in range(num_epochs):
        fusion_model.train()
        start = time.time()
        for step, batch in enumerate(dataloader):
            wavlm = batch["wavlm_embeddings"].to(device, non_blocking=True)
            hubert = batch["w2v2_embeddings"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True).long()
            target_lengths = batch["target_lengths"].to(device, non_blocking=True).long()
            input_lengths  = batch["input_lengths"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                logits = fusion_model(wavlm, hubert)               # [B, T*, V]
                log_probs = F.log_softmax(logits, dim=-1).to(torch.float32).transpose(0, 1)  # [T*, B, V]
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % 200 == 0 or (step + 1) == num_batches:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log({"train/loss": loss.item(), "train/lr": lr_now,
                           "epoch": epoch + (step + 1)/num_batches})
                print(f"  Epoch {epoch+1}, Step {step+1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, LR: {lr_now:.2e}")

        print(f"Epoch {epoch+1}/{num_epochs}, Last Loss: {loss.item():.4f}")
        print(f"Time for epoch {epoch+1}: {time.time()-start:.2f}s")

        # ---- Eval ----
        eval_wer = eval_fn()
        print(f"Evaluation WER after epoch {epoch+1}: {eval_wer:.4f}")
        wandb.log({"eval/wer": eval_wer, "epoch": epoch + 1})

        if eval_wer < best_wer:
            best_wer = eval_wer
            os.makedirs(save_dir, exist_ok=True)
            ckpt = os.path.join(save_dir, "fusion_best.pt")
            torch.save(fusion_model.state_dict(), ckpt)
            print(f"New best WER {best_wer:.4f}. Model saved to {ckpt}")

    wandb.finish()