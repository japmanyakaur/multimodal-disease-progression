"""Neural models.

ClinicalEncoder : GRU over the visit history (Eq. 6.1)
MRIEncoder      : small MLP over MRI measures; each measure is also a "token"
                  for cross-attention (Eq. 6.2)
FusionModel     : one class, five modes
    clinical  - E2, clinical history only
    mri       - E3, MRI only
    early     - E4, concatenate both representations
    xattn     - E4, cross-attention without a reliability gate
    proposed  - E5, cross-attention + reliability gate (Eq. 6.3-6.5) + modality dropout (Eq. 6.7)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClinicalEncoder(nn.Module):
    def __init__(self, n_feat, n_static, hidden, use_time=True, use_mask=True, dropout=0.2):
        super().__init__()
        self.use_time, self.use_mask = use_time, use_mask
        in_dim = n_feat
        if use_mask:
            in_dim += n_feat
        if use_time:
            in_dim += (n_feat if use_mask else 0) + 2
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, b):
        parts = [b["x"]]
        if self.use_mask:
            parts.append(b["m"])
        if self.use_time:
            if self.use_mask:                  # per-score "time since measured" reveals missingness
                parts.append(b["dt"])
            parts += [b["gap"], b["tti"]]
        inp = torch.cat(parts, dim=-1)
        L = inp.shape[1]
        packed = pack_padded_sequence(inp, b["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.gru(packed)
        states, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        pad = torch.arange(L)[None, :] >= b["lengths"][:, None]
        h = self.out(torch.cat([h_n[-1], b["static"]], dim=-1))
        return h, states, pad


class MRIEncoder(nn.Module):
    """kind="roi": MRI measures, one token per measure.  kind="cnn": frozen 3D-CNN embedding + 4x4x4 feature-map tokens."""

    def __init__(self, n_roi, hidden, dropout=0.2, kind="roi", emb_dim=64, n_tokens=64):
        super().__init__()
        self.kind = kind
        n_in = n_roi if kind == "roi" else emb_dim
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        if kind == "roi":
            self.value_weight = nn.Parameter(torch.randn(n_roi, hidden) * 0.1)
            self.region_embedding = nn.Parameter(torch.randn(n_roi, hidden) * 0.1)
        else:
            self.token_proj = nn.Linear(emb_dim, hidden)
            self.position = nn.Parameter(torch.randn(n_tokens, hidden) * 0.1)

    def forward(self, b):
        if self.kind == "roi":
            roi = b["roi"]
            tokens = roi.unsqueeze(-1) * self.value_weight + self.region_embedding   # (B, regions, hidden)
            return self.mlp(roi), tokens
        return self.mlp(b["img_emb"]), self.token_proj(b["img_tokens"]) + self.position


class FusionModel(nn.Module):
    def __init__(self, mode, n_feat, n_static, n_roi, n_rel, hidden=32, dropout=0.2, heads=2,
                 use_gate=True, use_xattn=True, use_time=True, use_mask=True,
                 p_drop_mri=0.0, p_drop_clinical=0.0, mri_kind="roi"):
        super().__init__()
        assert mode in ("clinical", "mri", "early", "xattn", "proposed")
        self.mode, self.use_gate, self.use_xattn = mode, use_gate, use_xattn
        self.p_drop_mri, self.p_drop_clinical = p_drop_mri, p_drop_clinical
        if mode != "mri":
            self.clin = ClinicalEncoder(n_feat, n_static, hidden, use_time, use_mask, dropout)
        if mode != "clinical":
            self.mri = MRIEncoder(n_roi, hidden, dropout, kind=mri_kind)
        if mode in ("xattn", "proposed") and use_xattn:
            self.att_clin = nn.MultiheadAttention(hidden, heads, batch_first=True)   # clinical queries MRI tokens
            self.att_mri = nn.MultiheadAttention(hidden, heads, batch_first=True)    # MRI queries visit states
            self.norm_clin, self.norm_mri = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        if mode == "proposed" and use_gate:
            self.reliability = nn.Sequential(nn.Linear(n_rel, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        head_in = {"clinical": hidden, "mri": hidden, "early": 2 * hidden + 2,
                   "xattn": 2 * hidden + 2, "proposed": hidden}[mode]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def availability(self, b, moddrop=True):
        """Modality dropout (Eq. 6.7): during training randomly hide whole modalities, keep >= 1."""
        a_c, a_m = b["a_clin"], b["a_mri"]
        if not (self.training and moddrop) or (self.p_drop_mri == 0 and self.p_drop_clinical == 0):
            return a_c, a_m
        d_m = a_m * (torch.rand_like(a_m) >= self.p_drop_mri).float()
        d_c = a_c * (torch.rand_like(a_c) >= self.p_drop_clinical).float()
        d_c = torch.where((d_c + d_m) == 0, a_c, d_c)
        d_m = torch.where((d_c + d_m) == 0, a_m, d_m)
        return d_c, d_m

    def forward(self, b, moddrop=True):
        a_c, a_m = self.availability(b, moddrop)
        extras = {}
        if self.mode == "clinical":
            return self.head(self.clin(b)[0]).squeeze(-1), extras
        if self.mode == "mri":
            return self.head(self.mri(b)[0]).squeeze(-1), extras

        h_c, states, pad = self.clin(b)
        h_m, tokens = self.mri(b)
        if self.mode == "early":
            z = torch.cat([h_c * a_c[:, None], h_m * a_m[:, None], a_c[:, None], a_m[:, None]], dim=-1)
            return self.head(z).squeeze(-1), extras

        if self.use_xattn:                                     # Eq. 6.4
            c_att, _ = self.att_clin(h_c[:, None], tokens, tokens)
            m_att, _ = self.att_mri(h_m[:, None], states, states, key_padding_mask=pad)
            c_new = self.norm_clin(h_c + c_att[:, 0])
            m_new = self.norm_mri(h_m + m_att[:, 0])
            if self.mode == "proposed":                        # bypass attention if a modality is absent
                both = ((a_c * a_m) > 0)[:, None]
                c_new, m_new = torch.where(both, c_new, h_c), torch.where(both, m_new, h_m)
            h_c, h_m = c_new, m_new

        if self.mode == "xattn":
            z = torch.cat([h_c, h_m, a_c[:, None], a_m[:, None]], dim=-1)
            return self.head(z).squeeze(-1), extras

        avail = torch.stack([a_c, a_m], dim=1)
        if self.use_gate:                                      # Eq. 6.3: availability-masked softmax
            rel = b["rel"].clone()
            rel[:, 0], rel[:, 1] = a_c, a_m
            scores = self.reliability(rel).masked_fill(avail == 0, -1e9)
            alpha = torch.softmax(scores, dim=1)
        else:
            alpha = avail / avail.sum(dim=1, keepdim=True)
        z = alpha[:, :1] * h_c + alpha[:, 1:] * h_m            # Eq. 6.5
        extras["alpha"] = alpha
        return self.head(z).squeeze(-1), extras


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
