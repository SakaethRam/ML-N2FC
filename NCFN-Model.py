
# ============================================================
# Block 6 — Full NCFN Model
# ============================================================


def _make_classification_head(
    in_dim: int, out_classes: int, dropout: float
) -> nn.Sequential:
    """Standard classification head: Linear → GELU → Dropout → Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim // 2, out_classes),
    )


def _make_regression_head(
    in_dim: int, dropout: float
) -> nn.Sequential:
    """Standard regression head: Linear → GELU → Dropout → Linear(1)."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim // 4, 1),
    )


class NCFNModel(nn.Module):
    """
    Neural Context Fusion Network — full multimodal model.

    Architecture:
        SpeechEncoder   ─┐
        GameplayEncoder  ├─► CrossModalAttentionFusion ─► Multi-task Heads
        ChatEncoder     ─┘

    Outputs:
        persona_logits  : [B, NUM_PERSONAS]
        emotion_logits  : [B, NUM_EMOTIONS]
        pitch_pred      : [B, 1]
        rate_pred       : [B, 1]
        style_logits    : [B, NUM_STYLES]
    """

    def __init__(self, config: NCFNConfig):
        super().__init__()
        self.config = config

        # ── Encoders ─────────────────────────────────────────
        self.speech_encoder   = SpeechEncoder(config)
        self.gameplay_encoder = GameplayEncoder(config)
        self.chat_encoder     = ChatEncoder(config)

        # ── Fusion Layer ──────────────────────────────────────
        self.fusion = CrossModalAttentionFusion(config)

        # ── Multi-task Heads ──────────────────────────────────
        D = config.FUSION_DIM
        drop = config.DROPOUT_RATE

        self.persona_head = _make_classification_head(D, config.NUM_PERSONAS, drop)
        self.emotion_head = _make_classification_head(D, config.NUM_EMOTIONS, drop)
        self.pitch_head   = _make_regression_head(D, drop)
        self.rate_head    = _make_regression_head(D, drop)
        self.style_head   = _make_classification_head(D, config.NUM_STYLES, drop)

    def forward(
        self,
        speech:   torch.Tensor,   # [B, SPEECH_EMBED_DIM]
        gameplay: torch.Tensor,   # [B, GAMEPLAY_FEATURE_DIM]
        chat:     torch.Tensor,   # [B, CHAT_EMBED_DIM]
    ) -> Dict[str, torch.Tensor]:

        # ── Encode each modality ─────────────────────────────
        s = self.speech_encoder(speech)       # [B, FUSION_DIM // 3]
        g = self.gameplay_encoder(gameplay)   # [B, FUSION_DIM // 3]
        c = self.chat_encoder(chat)           # [B, FUSION_DIM // 3]

        # ── Fuse modalities ───────────────────────────────────
        fused = self.fusion(s, g, c)          # [B, FUSION_DIM]

        # ── Multi-task predictions ────────────────────────────
        persona_logits = self.persona_head(fused)   # [B, NUM_PERSONAS]
        emotion_logits = self.emotion_head(fused)   # [B, NUM_EMOTIONS]
        pitch_pred     = self.pitch_head(fused)     # [B, 1]
        rate_pred      = self.rate_head(fused)      # [B, 1]
        style_logits   = self.style_head(fused)     # [B, NUM_STYLES]

        return {
            "persona_logits": persona_logits,
            "emotion_logits": emotion_logits,
            "pitch_pred":     pitch_pred,
            "rate_pred":      rate_pred,
            "style_logits":   style_logits,
        }


# ─── Instantiate Model ────────────────────────────────────────

model = NCFNModel(cfg).to(cfg.DEVICE)

total_params = count_parameters(model)
encoder_params = (
    count_parameters(model.speech_encoder)
    + count_parameters(model.gameplay_encoder)
    + count_parameters(model.chat_encoder)
)
fusion_params = count_parameters(model.fusion)
head_params   = (
    count_parameters(model.persona_head)
    + count_parameters(model.emotion_head)
    + count_parameters(model.pitch_head)
    + count_parameters(model.rate_head)
    + count_parameters(model.style_head)
)

print("=" * 60)
print("  Full NCFN Model Summary")
print("=" * 60)
print(f"\n[Architecture]")
print(f"  Encoder params    : {encoder_params:>10,}")
print(f"  Fusion params     : {fusion_params:>10,}")
print(f"  Head params       : {head_params:>10,}")
print(f"  ─────────────────────────────")
print(f"  TOTAL params      : {total_params:>10,}")
print(f"  Device            : {cfg.DEVICE}")

# ─── Forward Pass Shape Verification ─────────────────────────

_B = 8  # test batch size
_dummy_speech   = torch.randn(_B, cfg.SPEECH_EMBED_DIM).to(cfg.DEVICE)
_dummy_gameplay = torch.randn(_B, cfg.GAMEPLAY_FEATURE_DIM).to(cfg.DEVICE)
_dummy_chat     = torch.randn(_B, cfg.CHAT_EMBED_DIM).to(cfg.DEVICE)

model.eval()
with torch.no_grad():
    _outputs = model(_dummy_speech, _dummy_gameplay, _dummy_chat)

print(f"\n[Forward Pass — batch size {_B}]")
print(f"  persona_logits : {tuple(_outputs['persona_logits'].shape)}"
      f"  (expected ({_B}, {cfg.NUM_PERSONAS}))")
print(f"  emotion_logits : {tuple(_outputs['emotion_logits'].shape)}"
      f"  (expected ({_B}, {cfg.NUM_EMOTIONS}))")
print(f"  pitch_pred     : {tuple(_outputs['pitch_pred'].shape)}"
      f"  (expected ({_B}, 1))")
print(f"  rate_pred      : {tuple(_outputs['rate_pred'].shape)}"
      f"  (expected ({_B}, 1))")
print(f"  style_logits   : {tuple(_outputs['style_logits'].shape)}"
      f"  (expected ({_B}, {cfg.NUM_STYLES}))")

model.train()  # reset to train mode for the training block
print("\n[Block 6] Full NCFN model — DONE")
