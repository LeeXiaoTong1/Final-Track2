from pathlib import Path
import re
import shutil

p = Path("model.py")
bak = Path("model.py.bak_before_stable_cross")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[backup] {p} -> {bak}")

s = p.read_text(encoding="utf-8")

new_class = r'''class CrossStreamFusionBlock(nn.Module):
    """
    Stable gated semantic-forensic cross interaction.

    This block replaces the original bidirectional MultiheadAttention version.
    It keeps --ufm_layers > 0 meaningful, but avoids attention-backward NaN.

    Input:
      semantic_tokens: [B, T, D]
      forensic_tokens: [B, T, D]

    Output:
      semantic_tokens, forensic_tokens with bounded cross-stream residual update.
    """
    def __init__(
        self,
        dim=512,
        num_heads=8,   # kept for compatibility with old constructor; not used
        dropout=0.1
    ):
        super(CrossStreamFusionBlock, self).__init__()

        self.dim = dim

        self.sem_norm = nn.LayerNorm(dim)
        self.for_norm = nn.LayerNorm(dim)

        # Global context extraction: mean + max pooling.
        self.sem_ctx_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.for_ctx_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Forensic context modulates semantic tokens.
        self.for_to_sem_delta = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.for_to_sem_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )

        # Semantic context modulates forensic tokens.
        self.sem_to_for_delta = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.sem_to_for_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )

        # Small residual scales. Start close to identity.
        self.raw_cross_scale = nn.Parameter(torch.tensor(-3.0))
        self.raw_ffn_scale = nn.Parameter(torch.tensor(-3.0))

        self.sem_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self.for_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self._stable_init()

    def _stable_init(self):
        # Delta branches initially output zero, so the block starts as identity.
        nn.init.zeros_(self.for_to_sem_delta[-1].weight)
        nn.init.zeros_(self.for_to_sem_delta[-1].bias)
        nn.init.zeros_(self.sem_to_for_delta[-1].weight)
        nn.init.zeros_(self.sem_to_for_delta[-1].bias)

        nn.init.zeros_(self.sem_ffn[-1].weight)
        nn.init.zeros_(self.sem_ffn[-1].bias)
        nn.init.zeros_(self.for_ffn[-1].weight)
        nn.init.zeros_(self.for_ffn[-1].bias)

        # Gates start small.
        nn.init.zeros_(self.for_to_sem_gate[-1].weight)
        nn.init.constant_(self.for_to_sem_gate[-1].bias, -2.0)
        nn.init.zeros_(self.sem_to_for_gate[-1].weight)
        nn.init.constant_(self.sem_to_for_gate[-1].bias, -2.0)

    def _pool(self, x):
        # x: [B, T, D]
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def _check_finite(self, name, x):
        if not torch.isfinite(x).all():
            with torch.no_grad():
                y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"[NONFINITE STABLE CROSS] {name} | "
                    f"shape={tuple(x.shape)} | "
                    f"nan={torch.isnan(x).any().item()} | "
                    f"inf={torch.isinf(x).any().item()} | "
                    f"min={y.min().item():.4e} | "
                    f"max={y.max().item():.4e}"
                )
            raise RuntimeError(f"Non-finite stable cross tensor: {name}")
        return x

    def forward(self, semantic_tokens, forensic_tokens):
        sem = self.sem_norm(torch.clamp(semantic_tokens, -20.0, 20.0))
        forg = self.for_norm(torch.clamp(forensic_tokens, -20.0, 20.0))

        sem_pool = self._pool(sem)
        for_pool = self._pool(forg)

        sem_ctx = self.sem_ctx_proj(sem_pool).unsqueeze(1).expand(-1, sem.size(1), -1)
        for_ctx = self.for_ctx_proj(for_pool).unsqueeze(1).expand(-1, forg.size(1), -1)

        sem_in = torch.cat([sem, for_ctx], dim=-1)
        for_in = torch.cat([forg, sem_ctx], dim=-1)

        sem_delta = self.for_to_sem_delta(sem_in)
        for_delta = self.sem_to_for_delta(for_in)

        sem_gate = torch.sigmoid(self.for_to_sem_gate(sem_in))
        for_gate = torch.sigmoid(self.sem_to_for_gate(for_in))

        sem_delta = self._check_finite("sem_delta", sem_delta)
        for_delta = self._check_finite("for_delta", for_delta)

        sem_delta = torch.clamp(sem_delta, -5.0, 5.0)
        for_delta = torch.clamp(for_delta, -5.0, 5.0)

        # Strictly bounded residual strength.
        cross_alpha = 0.05 * torch.sigmoid(self.raw_cross_scale)

        semantic_tokens = semantic_tokens + cross_alpha * sem_gate * sem_delta
        forensic_tokens = forensic_tokens + cross_alpha * for_gate * for_delta

        sem_ffn = self.sem_ffn(semantic_tokens)
        for_ffn = self.for_ffn(forensic_tokens)

        sem_ffn = self._check_finite("sem_ffn", sem_ffn)
        for_ffn = self._check_finite("for_ffn", for_ffn)

        sem_ffn = torch.clamp(sem_ffn, -5.0, 5.0)
        for_ffn = torch.clamp(for_ffn, -5.0, 5.0)

        ffn_alpha = 0.05 * torch.sigmoid(self.raw_ffn_scale)

        semantic_tokens = semantic_tokens + ffn_alpha * sem_ffn
        forensic_tokens = forensic_tokens + ffn_alpha * for_ffn

        semantic_tokens = self._check_finite("semantic_tokens_out", semantic_tokens)
        forensic_tokens = self._check_finite("forensic_tokens_out", forensic_tokens)

        return semantic_tokens, forensic_tokens

'''

pattern = r"(?ms)^class CrossStreamFusionBlock\(nn\.Module\):.*?(?=^class BiCrossStreamTransformer\(nn\.Module\):)"
s2, n = re.subn(pattern, new_class + "\n", s, count=1)

if n != 1:
    raise RuntimeError(f"Failed to replace CrossStreamFusionBlock. matched={n}")

p.write_text(s2, encoding="utf-8")
print("[done] CrossStreamFusionBlock replaced by stable gated version")
