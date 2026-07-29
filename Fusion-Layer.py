
# ============================================================
# Block 5 — Cross-Modal Attention Fusion Layer
# ============================================================


class CrossModalAttentionFusion(nn.Module):
    """
    Fuses speech, gameplay, and chat feature vectors via cross-modal
    multi-head self-attention in a transformer-style pre-norm block.

    Input : 3 vectors each of shape [B, FUSION_DIM // 3]
    Output: [B, FUSION_DIM]
    """

    def __init__(self, config: NCFNConfig):
        super().__init__()

        D_token  = config.FUSION_DIM // 3      # token dimension (each modality)
        D_fusion = config.FUSION_DIM            # final output dimension
        n_heads  = config.NUM_ATTENTION_HEADS
        drop     = config.DROPOUT_RATE

        # Pre-norm layers (one per modality token before attention)
        self.pre_norm_speech   = nn.LayerNorm(D_token)
        self.pre_norm_gameplay = nn.LayerNorm(D_token)
        self.pre_norm_chat     = nn.LayerNorm(D_token)

        # Multi-head self-attention over the 3-token sequence
        # Requires embed_dim divisible by num_heads
        assert D_token % n_heads == 0, (
            f"D_token ({D_token}) must be divisible by NUM_ATTENTION_HEADS ({n_heads}). "
            f"FUSION_DIM // 3 = {D_token}. Adjust FUSION_DIM or NUM_ATTENTION_HEADS."
        )
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=D_token,
            num_heads=n_heads,
            dropout=drop,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(drop)
        self.post_attn_norm = nn.LayerNorm(D_token)

        # Position-wise FFN (applied per token)
        self.ffn = nn.Sequential(
            nn.Linear(D_token, D_token * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(D_token * 4, D_token),
            nn.Dropout(drop),
        )
        self.post_ffn_norm = nn.LayerNorm(D_token)

        # Final projection: [B, 3 * D_token] → [B, FUSION_DIM]
        self.out_proj = nn.Sequential(
            nn.Linear(3 * D_token, D_fusion),
            nn.LayerNorm(D_fusion),
            nn.GELU(),
        )

    def forward(
        self,
        speech_feat:   torch.Tensor,   # [B, D_token]
        gameplay_feat: torch.Tensor,   # [B, D_token]
        chat_feat:     torch.Tensor,   # [B, D_token]
    ) -> torch.Tensor:                 # [B, FUSION_DIM]

        B = speech_feat.size(0)

        # Pre-norm each modality token
        s = self.pre_norm_speech(speech_feat)      # [B, D_token]
        g = self.pre_norm_gameplay(gameplay_feat)  # [B, D_token]
        c = self.pre_norm_chat(chat_feat)          # [B, D_token]

        # Stack as sequence: [B, 3, D_token]
        tokens = torch.stack([s, g, c], dim=1)

        # Multi-head self-attention
        attn_out, _attn_weights = self.cross_modal_attn(tokens, tokens, tokens)
        attn_out = self.attn_dropout(attn_out)

        # Add & Norm (residual from pre-normed tokens)
        tokens = self.post_attn_norm(tokens + attn_out)     # [B, 3, D_token]

        # Position-wise FFN with residual
        ffn_out = self.ffn(tokens)                           # [B, 3, D_token]
        tokens = self.post_ffn_norm(tokens + ffn_out)        # [B, 3, D_token]

        # Flatten and project to FUSION_DIM
        flat = tokens.reshape(B, -1)                         # [B, 3 * D_token]
        fused = self.out_proj(flat)                          # [B, FUSION_DIM]
        return fused


# ─── Instantiate and Inspect ─────────────────────────────────

fusion_layer = CrossModalAttentionFusion(cfg)

print("=" * 60)
print("  Cross-Modal Attention Fusion Layer")
print("=" * 60)
print(f"\n[Architecture]")
print(fusion_layer)
print(f"\n[Dimensions]")
print(f"  D_token  (per-modality) : {cfg.FUSION_DIM // 3}")
print(f"  Num attention heads     : {cfg.NUM_ATTENTION_HEADS}")
print(f"  Output FUSION_DIM       : {cfg.FUSION_DIM}")
print(f"\n[Params]: {count_parameters(fusion_layer):,}")

# Quick forward pass check
_D_token = cfg.FUSION_DIM // 3
_s_feat  = torch.randn(4, _D_token)
_g_feat  = torch.randn(4, _D_token)
_c_feat  = torch.randn(4, _D_token)
_fused   = fusion_layer(_s_feat, _g_feat, _c_feat)
print(f"\n[Shape Check]")
print(f"  Input tokens : 3 × (4, {_D_token})")
print(f"  Fused output : {tuple(_fused.shape)}  (expected (4, {cfg.FUSION_DIM}))")
print("\n[Block 5] Cross-modal attention fusion layer — DONE")
