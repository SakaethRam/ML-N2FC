
# ============================================================
# Block 4 — Input Encoder Modules
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --- Speech Encoder ------------------------------------------

class SpeechEncoder(nn.Module):
    """
    Encodes speech embeddings (SPEECH_EMBED_DIM) -> FUSION_DIM // 3.
    Uses MLP with LayerNorm+GELU+Dropout and a residual connection.
    """

    def __init__(self, config):
        super().__init__()
        D_in  = config.SPEECH_EMBED_DIM
        D_mid = 256
        D_out = config.FUSION_DIM // 3
        drop  = config.DROPOUT_RATE

        # Input projection for residual (D_in -> D_out)
        self.residual_proj = nn.Linear(D_in, D_out, bias=False)

        # Main MLP: D_in -> 256 -> 256 -> D_out
        self.layer1 = nn.Sequential(
            nn.Linear(D_in, D_mid),
            nn.LayerNorm(D_mid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(D_mid, D_mid),
            nn.LayerNorm(D_mid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.layer3 = nn.Linear(D_mid, D_out)

        # Post-residual norm
        self.out_norm = nn.LayerNorm(D_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, SPEECH_EMBED_DIM]
        residual = self.residual_proj(x)           # [B, D_out]
        h = self.layer1(x)                          # [B, 256]
        h = self.layer2(h)                          # [B, 256]
        h = self.layer3(h)                          # [B, D_out]
        out = self.out_norm(h + residual)           # [B, D_out]
        return out


# --- Gameplay Encoder ----------------------------------------

class GameplayEncoder(nn.Module):
    """
    Encodes gameplay telemetry (GAMEPLAY_FEATURE_DIM) -> FUSION_DIM // 3.
    Includes a 1D self-attention layer over feature tokens.
    """

    def __init__(self, config):
        super().__init__()
        D_in  = config.GAMEPLAY_FEATURE_DIM
        D_mid = 128
        D_mid2 = 256
        D_out = config.FUSION_DIM // 3
        drop  = config.DROPOUT_RATE
        n_heads = config.NUM_ATTENTION_HEADS

        # Stage 1: D_in -> D_mid -> D_mid2 (standard MLP path)
        self.mlp1 = nn.Sequential(
            nn.Linear(D_in, D_mid),
            nn.LayerNorm(D_mid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(D_mid, D_mid2),
            nn.LayerNorm(D_mid2),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # Stage 2: Self-attention treating each gameplay feature group as a token
        NUM_FEATURE_TOKENS = 8          # 8 tokens, each of dim 32
        FEAT_TOKEN_DIM = D_mid2 // NUM_FEATURE_TOKENS  # 32
        assert D_mid2 % NUM_FEATURE_TOKENS == 0, "D_mid2 must be divisible by NUM_FEATURE_TOKENS"

        self.NUM_FEATURE_TOKENS = NUM_FEATURE_TOKENS
        self.FEAT_TOKEN_DIM = FEAT_TOKEN_DIM

        self.self_attn = nn.MultiheadAttention(
            embed_dim=FEAT_TOKEN_DIM,
            num_heads=min(n_heads, FEAT_TOKEN_DIM // 8),   # heads must divide dim
            dropout=drop,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(FEAT_TOKEN_DIM)

        # Stage 3: Flatten + project to D_out
        self.out_proj = nn.Sequential(
            nn.Linear(D_mid2, D_out),
            nn.LayerNorm(D_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, GAMEPLAY_FEATURE_DIM]
        h = self.mlp1(x)                                                  # [B, D_mid]
        h = self.mlp2(h)                                                   # [B, D_mid2]

        # Reshape to token sequence
        B = h.size(0)
        h_tokens = h.view(B, self.NUM_FEATURE_TOKENS, self.FEAT_TOKEN_DIM)  # [B, T, D_token]

        # Self-attention
        h_attn, _ = self.self_attn(h_tokens, h_tokens, h_tokens)           # [B, T, D_token]
        h_attn = self.attn_norm(h_attn + h_tokens)                          # residual + norm

        # Flatten back and project
        h_flat = h_attn.reshape(B, -1)                                      # [B, D_mid2]
        out = self.out_proj(h_flat)                                          # [B, D_out]
        return out


# --- Chat Encoder ---------------------------------------------

class ChatEncoder(nn.Module):
    """
    Encodes chat & sentiment features (CHAT_EMBED_DIM) -> FUSION_DIM // 3.
    Includes a sigmoid gate on the toxicity/hype component (first 3 dims).
    """

    def __init__(self, config):
        super().__init__()
        D_in  = config.CHAT_EMBED_DIM
        D_mid = 128
        D_mid2 = 256
        D_out = config.FUSION_DIM // 3
        drop  = config.DROPOUT_RATE

        # Gate branch: operate on the named scalar dims (sentiment, hype, toxicity = 3 dims)
        GATE_DIMS = 3
        self.gate_linear = nn.Linear(GATE_DIMS, GATE_DIMS)

        # Main MLP
        self.layer1 = nn.Sequential(
            nn.Linear(D_in, D_mid),
            nn.LayerNorm(D_mid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(D_mid, D_mid2),
            nn.LayerNorm(D_mid2),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(D_mid2, D_out),
            nn.LayerNorm(D_out),
            nn.GELU(),
        )

        # Gate modulation: scale the output from MLP stage 1 using gate signal
        self.gate_proj = nn.Linear(GATE_DIMS, D_mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, CHAT_EMBED_DIM]
        # Extract hype/toxicity/sentiment gate signal (indices 0,1,2)
        gate_input = x[:, :3]                                    # [B, 3]
        gate_signal = torch.sigmoid(self.gate_linear(gate_input))  # [B, 3] in [0,1]

        # Main MLP stage 1
        h = self.layer1(x)                                         # [B, D_mid]

        # Modulate with gate (broadcast gate to D_mid via a learned projection)
        gate_mod = torch.sigmoid(self.gate_proj(gate_signal))      # [B, D_mid]
        h = h * gate_mod                                            # element-wise gating

        # Stages 2 & 3
        h = self.layer2(h)                                          # [B, D_mid2]
        out = self.layer3(h)                                        # [B, D_out]
        return out


# --- Instantiate and Print Param Counts -----------------------

speech_encoder   = SpeechEncoder(cfg)
gameplay_encoder = GameplayEncoder(cfg)
chat_encoder     = ChatEncoder(cfg)

print("=" * 60)
print("  Input Encoder Modules Summary")
print("=" * 60)
print(f"\n[SpeechEncoder]")
print(f"  Input  : {cfg.SPEECH_EMBED_DIM}  ->  Output: {cfg.FUSION_DIM // 3}")
print(f"  Params : {count_parameters(speech_encoder):,}")

print(f"\n[GameplayEncoder]")
print(f"  Input  : {cfg.GAMEPLAY_FEATURE_DIM}  ->  Output: {cfg.FUSION_DIM // 3}")
print(f"  Params : {count_parameters(gameplay_encoder):,}")

print(f"\n[ChatEncoder]")
print(f"  Input  : {cfg.CHAT_EMBED_DIM}  ->  Output: {cfg.FUSION_DIM // 3}")
print(f"  Params : {count_parameters(chat_encoder):,}")

print(f"\n[Total Encoder Params]: "
      f"{count_parameters(speech_encoder) + count_parameters(gameplay_encoder) + count_parameters(chat_encoder):,}")

# Quick shape verification
_bs = 4
_s = torch.randn(_bs, cfg.SPEECH_EMBED_DIM)
_g = torch.randn(_bs, cfg.GAMEPLAY_FEATURE_DIM)
_c = torch.randn(_bs, cfg.CHAT_EMBED_DIM)

_s_out = speech_encoder(_s)
_g_out = gameplay_encoder(_g)
_c_out = chat_encoder(_c)
print(f"\n[Shape Check]")
print(f"  SpeechEncoder   output: {tuple(_s_out.shape)}  (expected ({_bs}, {cfg.FUSION_DIM // 3}))")
print(f"  GameplayEncoder output: {tuple(_g_out.shape)}  (expected ({_bs}, {cfg.FUSION_DIM // 3}))")
print(f"  ChatEncoder     output: {tuple(_c_out.shape)}  (expected ({_bs}, {cfg.FUSION_DIM // 3}))")
print("\n[Block 4] Input encoder modules -- DONE")
