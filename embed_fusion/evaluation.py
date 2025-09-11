import os, time, torch, evaluate
from torch.amp import autocast
from english_normalizer import EnglishTextNormalizer

def run_evaluate(fusion_model, embedding_dir_wavlm, embedding_dir_w2v2,
                 text_file, processor, device):
    fusion_model.eval()
    normalizer = EnglishTextNormalizer({})
    data_list = []
    with open(text_file, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2: continue
            utt_id, transcript = parts
            p1 = os.path.join(embedding_dir_wavlm, f"{utt_id}_wavlm.pt")
            p2 = os.path.join(embedding_dir_w2v2,  f"{utt_id}_delta.pt")
            if os.path.exists(p1) and os.path.exists(p2):
                data_list.append({
                    "utt_id": utt_id,
                    "transcript": normalizer(transcript).lower(),
                    "wavlm": p1, "hubert": p2
                })

    metric_wer = evaluate.load("wer")
    start = time.time()
    with torch.no_grad():
        for item in data_list:
            wavlm = torch.load(item["wavlm"],  weights_only=True).to(device)
            hubert = torch.load(item["hubert"], weights_only=True).to(device)
            if wavlm.dim()==2: wavlm = wavlm.unsqueeze(0)
            if hubert.dim()==2: hubert = hubert.unsqueeze(0)

            T = min(wavlm.size(1), hubert.size(1))
            wavlm, hubert = wavlm[:, :T], hubert[:, :T]

            with autocast(device_type=device.type):
                logits = fusion_model(wavlm, hubert)           # [1, T, V]
                pred_ids = torch.argmax(logits, dim=-1).squeeze(0)
                pred_text = processor.decode(pred_ids)
            pred_text = normalizer(pred_text).lower()
            metric_wer.add_batch(predictions=[pred_text], references=[item["transcript"]])

    final_wer = float(metric_wer.compute())
    fusion_model.train()
    print(f"[EVAL] {len(data_list)} utts – time: {time.time()-start:.2f}s – WER: {final_wer:.4f}")
    return final_wer