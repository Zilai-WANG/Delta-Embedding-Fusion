# data.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Any
from english_normalizer import EnglishTextNormalizer

class FusionEmbeddingDataset(Dataset):
    """
    Loads two frame-aligned embedding streams per utterance.
    Stores each item's frame length T (min of two streams) for bucketed batching.
    """
    def __init__(self, embedding_dir_wavlm: str, embedding_dir_hubert: str, text_file: str, processor):
        self.processor = processor
        self.normalizer = EnglishTextNormalizer({})
        self.data: List[Dict[str, Any]] = []
        self.time_lengths: List[int] = []

        kept, skipped = 0, 0
        with open(text_file, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                utt_id, transcript = parts
                wavlm_path = os.path.join(embedding_dir_wavlm, f"{utt_id}_wavlm.pt")
                hubert_path  = os.path.join(embedding_dir_hubert,  f"{utt_id}_delta.pt")
                if not (os.path.exists(wavlm_path) and os.path.exists(hubert_path)):
                    skipped += 1
                    continue

                try:
                    a = torch.load(wavlm_path, map_location="cpu", weights_only=True)
                    b = torch.load(hubert_path,  map_location="cpu", weights_only=True)
                    if a.dim() == 3 and a.size(0) == 1: a = a.squeeze(0)
                    if b.dim() == 3 and b.size(0) == 1: b = b.squeeze(0)
                    if a.dim() != 2 or b.dim() != 2:
                        skipped += 1
                        continue
                    T = min(a.size(0), b.size(0))
                except Exception:
                    skipped += 1
                    continue

                self.data.append({
                    "utt_id": utt_id,
                    "transcript": self.normalizer(transcript).lower(),
                    "wavlm_path":  wavlm_path,
                    "hubert_path":   hubert_path,
                    "T": T
                })
                self.time_lengths.append(T)
                kept += 1

        self.time_lengths = np.asarray(self.time_lengths, dtype=np.int32)
        print(f"[Dataset] kept {kept} items, skipped {skipped}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wavlm_emb = torch.load(item["wavlm_path"], map_location="cpu", weights_only=True)
        hubert_emb  = torch.load(item["hubert_path"],  map_location="cpu", weights_only=True)
        if wavlm_emb.dim() == 3 and wavlm_emb.size(0) == 1: wavlm_emb = wavlm_emb.squeeze(0)
        if hubert_emb.dim()  == 3 and hubert_emb.size(0)  == 1: hubert_emb  = hubert_emb.squeeze(0)

        # Crop to same T so frame-wise fusion is aligned
        T = min(wavlm_emb.size(0), hubert_emb.size(0))
        wavlm_emb = wavlm_emb[:T]
        hubert_emb  = hubert_emb[:T]

        target = self.processor.tokenizer(item["transcript"], return_tensors="pt", padding=True).input_ids.squeeze(0)
        target_lengths = torch.tensor([target.size(0)], dtype=torch.long)

        return {
            "utt_id":            item["utt_id"],
            "wavlm_embedding":   wavlm_emb,    # [T, D]
            "hubert_embedding":    hubert_emb,     # [T, D]
            "target":            target,       # [L]
            "target_lengths":    target_lengths,
        }

def collate_fn(batch):
    import torch.nn as nn
    wavlm_list = [it["wavlm_embedding"] for it in batch]
    hubert_list  = [it["hubert_embedding"]  for it in batch]
    targets    = [it["target"]          for it in batch]
    targ_lens  = [it["target_lengths"].item() for it in batch]
    in_lens    = [emb.size(0) for emb in wavlm_list]
    utt_ids    = [it["utt_id"] for it in batch]

    wavlm_embeddings = nn.utils.rnn.pad_sequence(wavlm_list, batch_first=True)  # [B, T*, D]
    hubert_embeddings  = nn.utils.rnn.pad_sequence(hubert_list,  batch_first=True)
    targets          = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    import torch
    return {
        "utt_ids":          utt_ids,
        "wavlm_embeddings": wavlm_embeddings,
        "hubert_embeddings":  hubert_embeddings,
        "targets":          targets.long(),
        "target_lengths":   torch.tensor(targ_lens, dtype=torch.long),
        "input_lengths":    torch.tensor(in_lens,  dtype=torch.long),
    }

class BucketBatchSampler(Sampler):
    """
    Buckets indices by similar lengths; shuffles buckets and samples within
    buckets each epoch; yields lists of indices as batches.
    """
    def __init__(self, lengths, batch_size, num_buckets=30, shuffle=True, drop_last=False, seed=42):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.num_buckets = max(1, int(num_buckets))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        ranks = self.lengths.argsort()
        self.buckets = [sp.tolist() for sp in np.array_split(ranks, self.num_buckets)]

    def set_epoch(self, epoch:int):
        self.epoch = int(epoch)

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        buckets = list(self.buckets)
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i+self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total

def create_train_loader(paths: dict, hp: dict, processor, num_workers: int):
    dataset = FusionEmbeddingDataset(
        paths["train_embedding_dir_wavlm"],
        paths["train_embedding_dir_hubert"],
        paths["train_text_file"],
        processor
    )
    sampler = BucketBatchSampler(
        lengths=dataset.time_lengths,
        batch_size=hp["batch_size"],
        num_buckets=hp["num_buckets"],
        shuffle=True,
        drop_last=False,
        seed=hp["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader