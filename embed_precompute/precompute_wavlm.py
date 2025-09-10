import os
import argparse
import torch
import torchaudio
from transformers import WavLMForCTC, AutoFeatureExtractor

# Requires: pyyaml
#   pip install pyyaml
import yaml


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute WavLM hidden states from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file."
    )
    # Optional lightweight overrides (keep names aligned with YAML keys)
    parser.add_argument("--wavlm_model_path", type=str)
    parser.add_argument("--wavlm_processor_path", type=str)
    parser.add_argument("--wav_scp", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--layer_number", type=int)
    return parser.parse_args()


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    # Only override keys explicitly provided on the CLI
    for k in ["wavlm_model_path", "wavlm_processor_path", "wav_scp", "text", "out_dir", "device", "layer_number"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def build_utt_maps(wav_scp_path: str, text_path: str | None):
    # Parse wav.scp (Kaldi-style)
    utt_to_wav = {}
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, audio_cmd = parts
                # Handle the common "flac -c -d -s <path> |" command
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

    wavlm_out_dir = os.path.join(cfg["out_dir"], f"wavlm_embed_L{cfg['layer_number']}")
    os.makedirs(wavlm_out_dir, exist_ok=True)

    print(f"Layer number: {cfg['layer_number']}")
    print(f"Loading WavLM model from {cfg['wavlm_model_path']}")
    model_wavlm = WavLMForCTC.from_pretrained(cfg["wavlm_model_path"]).to(device)
    model_wavlm.eval()

    print("Loading feature extractor (no tokenizer needed)")
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg["wavlm_processor_path"])

    # Freeze model parameters.
    for p in model_wavlm.parameters():
        p.requires_grad = False

    # Build utt_id => wav_path (and optional text)
    utt_to_wav, utt_to_text = build_utt_maps(cfg["wav_scp"], cfg.get("text"))

    ############################
    # PRECOMPUTE + SAVE
    ############################
    with torch.no_grad():
        for utt_id, audio_path in utt_to_wav.items():
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except FileNotFoundError:
                print(f"File not found: {audio_path}")
                continue
            except Exception as e:
                print(f"Failed to load {audio_path}: {e}")
                continue

            # Convert to mono if necessary.
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed (use the extractor's expected rate).
            target_sr = getattr(feature_extractor, "sampling_rate", sample_rate)
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
                sample_rate = target_sr

            # Do NOT pad or truncate: use original waveform length.
            # Prepare model inputs.
            inputs_wavlm = feature_extractor(
                waveform.squeeze(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs_wavlm = to_device(inputs_wavlm, device)

            try:
                outputs_wavlm = model_wavlm(**inputs_wavlm, output_hidden_states=True, return_dict=True)
                out_wavlm = outputs_wavlm.hidden_states[cfg["layer_number"]]
            except Exception as e:
                print(f"Inference failed for {utt_id} ({audio_path}): {e}")
                continue

            # Save the embedding.
            wavlm_save = os.path.join(wavlm_out_dir, f"{utt_id}_wavlm.pt")
            torch.save(out_wavlm.cpu(), wavlm_save)

            # Clean up to free memory.
            del out_wavlm, outputs_wavlm, inputs_wavlm, waveform

    print(f"Done precomputing. Saved to {wavlm_out_dir}")


if __name__ == "__main__":
    main()