"""Temporal encoder baselines for the AG-EgoPose ablation.

These replace the Ego4D-pretrained ActionFormer temporal encoder with
generic temporal models trained *from scratch*. The point is to isolate
the action-pretrained prior from generic temporal modeling: if the
pretrained ActionFormer beats a matched-capacity TCN / bi-LSTM trained
from scratch, the gain comes from the action prior, not merely from
"having a temporal module" (Reviewer R2 W1, R4 W1).

Contract matches ActionFormerFeatureExtractor:
    forward(x, mask=None): (B, T, 384) -> (B, T, 384)

Both baselines are causal-optional so the same module answers R4's
streaming/causality concern too (pass causal=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNTemporalEncoder(nn.Module):
    """Dilated temporal convolutional network, trained from scratch.

    Matched to ActionFormer's ~20M-param scale is unnecessary and unfair
    (ActionFormer is mostly frozen; only ~few M adapt). We instead match
    the *trainable* capacity of the unfrozen ActionFormer path (~a few M)
    so the comparison is "same trainable budget, with vs without the
    pretrained action prior". Tune n_layers/hidden to hit the target.

    Dilations grow as 2^l, giving a receptive field of ~2*(2^n_layers - 1)+1
    frames — enough to cover the 64-frame window at n_layers>=6.
    """

    def __init__(self, dim=384, hidden=384, n_layers=6, kernel_size=3,
                 dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.in_proj = nn.Conv1d(dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for l in range(n_layers):
            dilation = 2 ** l
            self.blocks.append(_TCNBlock(hidden, kernel_size, dilation,
                                         dropout, causal))
        self.out_proj = nn.Conv1d(hidden, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x: (B, T, dim) -> (B, dim, T)
        B, T, _ = x.shape
        h = x.permute(0, 2, 1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_proj(h)          # (B, dim, T)
        h = h.permute(0, 2, 1)        # (B, T, dim)
        return self.out_norm(h)


class _TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout, causal):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.k = k
        # padding handled manually so causal is exact
        self.conv1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, ch)   # LayerNorm over channels
        self.norm2 = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)

    def _pad(self, x):
        pad = (self.k - 1) * self.dilation
        if self.causal:
            return F.pad(x, (pad, 0))       # left-only: no future leakage
        left = pad // 2
        return F.pad(x, (left, pad - left)) # centered: bidirectional

    def forward(self, x):
        residual = x
        h = self.conv1(self._pad(x))
        h = F.gelu(self.norm1(h))
        h = self.drop(h)
        h = self.conv2(self._pad(h))
        h = F.gelu(self.norm2(h))
        h = self.drop(h)
        return residual + h


class LSTMTemporalEncoder(nn.Module):
    """Bi-/uni-directional LSTM temporal encoder, trained from scratch.

    This is essentially the Lee et al. [16] style temporal module (an LSTM
    over per-frame features), which a reviewer cites directly. causal=True
    -> unidirectional (streaming-compatible), causal=False -> bidirectional.
    """

    def __init__(self, dim=384, hidden=384, n_layers=2, dropout=0.1,
                 causal=False):
        super().__init__()
        self.causal = causal
        bidirectional = not causal
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x: (B, T, dim)
        h, _ = self.lstm(x)           # (B, T, hidden*dirs)
        h = self.out_proj(h)          # (B, T, dim)
        return self.out_norm(h)


def build_temporal_baseline(kind, dim=384, causal=False):
    """Factory used by train.py / test.py.

    kind: 'tcn' | 'lstm'
    Returns a module with the ActionFormerFeatureExtractor forward contract.
    """
    if kind == 'tcn':
        return TCNTemporalEncoder(dim=dim, causal=causal)
    if kind == 'lstm':
        return LSTMTemporalEncoder(dim=dim, causal=causal)
    raise ValueError(f"unknown temporal baseline: {kind}")


if __name__ == '__main__':
    for kind in ['tcn', 'lstm']:
        for causal in [False, True]:
            m = build_temporal_baseline(kind, causal=causal)
            x = torch.randn(2, 64, 384)
            y = m(x)
            n = sum(p.numel() for p in m.parameters())
            assert y.shape == x.shape, (kind, causal, y.shape)
            print(f"{kind:5s} causal={causal!s:5s}  out={tuple(y.shape)}  params={n/1e6:.2f}M")
