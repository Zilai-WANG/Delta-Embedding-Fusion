# model.py
import torch
import torch.nn as nn

class FusionModelStage2(nn.Module):
    """
    Concat(WavLM, hubert) -> LayerNorm -> Dropout -> Linear(CTC vocab)
    """
    def __init__(self, input_dim_wavlm: int, input_dim_hubert: int, hidden_size: int, vocab_size: int):
        super().__init__()
        # hidden_size kept for interface parity (not used by this simple head)
        self.fusion_ln = nn.LayerNorm(input_dim_wavlm + input_dim_hubert)
        self.dropout = nn.Dropout(0.1)
        self.decoder = nn.Linear(input_dim_wavlm + input_dim_hubert, vocab_size)

    def forward(self, emb_wavlm: torch.Tensor, emb_hubert: torch.Tensor) -> torch.Tensor:
        # emb_*: [B, T, D]
        concat = torch.cat([emb_wavlm, emb_hubert], dim=-1)
        fused = self.fusion_ln(concat)
        fused = self.dropout(fused)
        logits = self.decoder(fused)  # [B, T, V]
        return logits