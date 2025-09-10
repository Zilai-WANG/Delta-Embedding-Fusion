import os
import argparse
import torch
import torchaudio
from transformers import HubertModel, AutoFeatureExtractor

# Requires: pyyaml
#   pip install pyyaml
import yaml


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute HuBERT delta features (finetuned - pretrained) from YAML.")
    parser.add_argument("--config", type=str, default="delta_config.yaml", help="Path to YAML config.")
    # Optional overrides (names match YAML keys)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--processor_path", type=str)
    parser.add_argument("--wav_scp", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--output_subdir_prefix", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--layer_number", type=int)
    return parser.parse_args()


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for k in [
        "pretrained_model_path",
        "finetuned_model_path",
        "processor_path",
        "wav_scp",
        "text",
        "out_dir",
        "output_subdir_prefix",
        "device",
        "layer_number",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def build_utt_maps(wav_scp_path: str, text_path: str | None):
    # Parse wav.scp (Kaldi-style). Handles common "flac -c -d -s <path> |" pattern.
    utt_to_wav = {}
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, audio_cmd = parts
                audio_path = audio_cmd.replace("flac -c -d -s ", "").rstrip(" |")
                utt_to_wav[utt_id] = audio_path

    utt_to_text = {}
    if text_path and os.path.exists(text_path):
        with open(text_path, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, transcript = parts
                    utt_to_text[utt_id] = transcript

    return utt_to_wav, utt_to_text


def to_device(batch_inputs, device: torch.device | str):
    # HF feature extractors return a dict of tensors
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_inputs.items()}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args)

    device = cfg["device"]

    # Output directory
    subdir = f"{cfg.get('output_subdir_prefix','hubert_delta_feature')}_L{cfg['layer_number']}"
    hubert_out_dir = os.path.join(cfg["out_dir"], subdir)
    os.makedirs(hubert_out_dir, exist_ok=True)

    print(f"Layer number: {cfg['layer_number']}")
    print(f"Loading PRETRAINED HuBERT from {cfg['pretrained_model_path']}")
    model_pre = HubertModel.from_pretrained(cfg["pretrained_model_path"]).to(device)
    model_pre.eval()

    print(f"Loading FINETUNED HuBERT from {cfg['finetuned_model_path']}")
    model_ft = HubertModel.from_pretrained(cfg["finetuned_model_path"]).to(device)
    model_ft.eval()

    print("Loading feature extractor (no tokenizer needed)")
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg["processor_path"])

    # Freeze model parameters.
    for p in model_pre.parameters():
        p.requires_grad = False
    for p in model_ft.parameters():
        p.requires_grad = False

    # Build maps
    utt_to_wav, utt_to_text = build_utt_maps(cfg["wav_scp"], cfg.get("text"))

    ############################
    # PRECOMPUTE + SAVE
    ############################
    with torch.no_grad():
        for utt_id, audio_path in utt_to_wav.items():
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except FileNotFoundError:
                print(f"[WARN] File not found: {audio_path}")
                continue
            except Exception as e:
                print(f"[WARN] Failed to load {audio_path}: {e}")
                continue

            # Convert to mono if necessary.
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to extractor's expected rate if needed.
            target_sr = getattr(feature_extractor, "sampling_rate", sample_rate)
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
                sample_rate = target_sr

            # Prepare inputs (no pad/truncate)
            inputs = feature_extractor(
                waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt"
            )
            inputs = to_device(inputs, device)

            try:
                outputs_pre = model_pre(**inputs, output_hidden_states=True, return_dict=True)
                outputs_ft  = model_ft(**inputs,  output_hidden_states=True, return_dict=True)

                emb_pre = outputs_pre.hidden_states[cfg["layer_number"]]   # [B, T, D]
                emb_ft  = outputs_ft.hidden_states[cfg["layer_number"]]    # [B, T, D]

                # delta = finetuned - pretrained
                delta_feat = emb_ft - emb_pre
            except Exception as e:
                print(f"[WARN] Inference failed for {utt_id} ({audio_path}): {e}")
                continue

            # Save delta embedding
            save_path = os.path.join(hubert_out_dir, f"{utt_id}_delta.pt")
            torch.save(delta_feat.cpu(), save_path)

            # Clean up
            del emb_pre, emb_ft, delta_feat, outputs_pre, outputs_ft, inputs, waveform
            torch.cuda.empty_cache()

    print(f"Done precomputing. Saved to {hubert_out_dir}")


if __name__ == "__main__":
    main()