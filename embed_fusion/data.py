import os
import torch
import torch.nn as nn
from english_normalizer import EnglishTextNormalizer

class FusionEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embedding_dir_wavlm, embedding_dir_w2v2, text_file, processor):
        self.processor = processor
        self.normalizer = EnglishTextNormalizer({})
        self.data = []
        kept, skipped = 0, 0
        with open(text_file, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1; continue
                utt_id, transcript = parts
                wavlm_path = os.path.join(embedding_dir_wavlm, f"{utt_id}_wavlm.pt")
                w2v2_path  = os.path.join(embedding_dir_w2v2,  f"{utt_id}_delta.pt")
                if not (os.path.exists(wavlm_path) and os.path.exists(w2v2_path)):
                    skipped += 1; continue
                self.data.append({
                    "utt_id": utt_id,
                    "transcript": self.normalizer(transcript).lower(),
                    "wavlm_path": wavlm_path,
                    "w2v2_path": w2v2_path
                })
                kept += 1
        print(f"[Dataset] kept {kept}, skipped {skipped}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wavlm_emb = torch.load(item["wavlm_path"], weights_only=True)
        w2v2_emb  = torch.load(item["w2v2_path"],  weights_only=True)
        if wavlm_emb.dim() == 3 and wavlm_emb.size(0) == 1: wavlm_emb = wavlm_emb.squeeze(0)
        if w2v2_emb.dim()  == 3 and w2v2_emb.size(0)  == 1: w2v2_emb  = w2v2_emb.squeeze(0)
        # Keep original T (no pad/truncation here)
        target = self.processor.tokenizer(item["transcript"], return_tensors="pt", padding=True).input_ids.squeeze(0)
        target_lengths = torch.tensor([target.size(0)], dtype=torch.long)
        return {
            "wavlm_embedding": wavlm_emb,   # [T, D]
            "w2v2_embedding":  w2v2_emb,    # [T, D]
            "target":          target,      # [L]
            "target_lengths":  target_lengths
        }

def collate_fn(batch):
    wavlm_list = [b["wavlm_embedding"] for b in batch]
    w2v2_list  = [b["w2v2_embedding"]  for b in batch]
    targets    = [b["target"]          for b in batch]
    targ_lens  = [b["target_lengths"].item() for b in batch]
    in_lens    = [x.size(0) for x in wavlm_list]

    wavlm_embeddings = nn.utils.rnn.pad_sequence(wavlm_list, batch_first=True)  # [B, T*, D]
    w2v2_embeddings  = nn.utils.rnn.pad_sequence(w2v2_list,  batch_first=True)
    # Use 0 padding (pad token / CTC blank id); target_lengths tell CTC the true size
    targets_pad      = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        "wavlm_embeddings": wavlm_embeddings,
        "w2v2_embeddings":  w2v2_embeddings,
        "targets":          targets_pad.long(),
        "target_lengths":   torch.tensor(targ_lens, dtype=torch.long),
        "input_lengths":    torch.tensor(in_lens,  dtype=torch.long),
    }