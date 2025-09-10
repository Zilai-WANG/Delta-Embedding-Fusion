# evaluation.py
import os
import time
import torch
import evaluate
from torch.amp import autocast
from typing import Tuple, List
from english_normalizer import EnglishTextNormalizer

def run_evaluate(
    fusion_model,
    embedding_dir_wavlm: str,
    embedding_dir_hubert: str,
    text_file: str,
    processor,
    device: torch.device,
    return_preds: bool = False
) -> Tuple[float, List]:
    """
    Returns: (wer, predictions) if return_preds=True, else (wer, None)
    predictions is a list of (utt_id, predicted_text)
    """
    fusion_model.eval()
    normalizer = EnglishTextNormalizer({})
    data_list = []

    with open(text_file, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, transcript = parts
                wavlm_path = os.path.join(embedding_dir_wavlm, f"{utt_id}_wavlm.pt")
                hubert_path  = os.path.join(embedding_dir_hubert,  f"{utt_id}_delta.pt")
                if os.path.exists(wavlm_path) and os.path.exists(hubert_path):
                    normalized_transcript = normalizer(transcript).lower()
                    data_list.append({
                        "utt_id": utt_id,
                        "transcript": normalized_transcript,
                        "wavlm_path": wavlm_path,
                        "hubert_path": hubert_path
                    })

    metric_wer = evaluate.load("wer")
    preds_out = [] if return_preds else None
    count = 0
    start_time_eval = time.time()

    with torch.no_grad():
        for item in data_list:
            wavlm_emb = torch.load(item["wavlm_path"], weights_only=True).to(device)
            hubert_emb  = torch.load(item["hubert_path"],  weights_only=True).to(device)
            if wavlm_emb.dim() == 2: wavlm_emb = wavlm_emb.unsqueeze(0)
            if hubert_emb.dim()  == 2: hubert_emb  = hubert_emb.unsqueeze(0)
            T = min(wavlm_emb.size(1), hubert_emb.size(1))
            wavlm_emb = wavlm_emb[:, :T]
            hubert_emb  = hubert_emb[:,  :T]

            with autocast(device_type=device.type):
                logits = fusion_model(wavlm_emb, hubert_emb)                 # [1, T, V]
                pred_ids = torch.argmax(logits, dim=-1)[:, :T]             # [1, T]
                pred_text = processor.batch_decode(pred_ids.cpu().numpy())[0]
                pred_text = normalizer(pred_text).lower()

            metric_wer.add_batch(predictions=[pred_text], references=[item["transcript"]])
            if return_preds:
                preds_out.append((item["utt_id"], pred_text))

            count += 1
            if count % 500 == 0:
                print(f"Evaluating on {count} samples")

    fusion_model.train()
    elapsed = time.time() - start_time_eval
    final_wer = float(metric_wer.compute())
    print(f"[EVAL] Completed on {len(data_list)} utterances – time: {elapsed:.2f}s – WER: {final_wer:.4f}")
    return final_wer, preds_out