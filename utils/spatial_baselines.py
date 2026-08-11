"""Single-network spatial baseline for the R5 (Expert) over-modularization ablation.

R5's concern: the spatial block stacks three submodules — a ConvNeXt-Tiny
heatmap net + a conv heatmap-tokenizer + a Spatial Joint Transformer — where
"a single network could tackle it." This module is that single network: one
ConvNeXt-Tiny trunk maps the image directly to J per-joint tokens, with NO
explicit 2D heatmap supervision, NO separate tokenizer, and NO inter-joint
transformer.

It is a drop-in producer of spatial_joint_features (B, T, J, dim), the same
tensor the 3-stage path feeds to the PoseDecoder. Comparing the two isolates
whether the staged decomposition (heatmap -> token -> inter-joint) actually
buys accuracy, or whether one net suffices (R5's hypothesis).

Fairness notes:
  * Same trunk (ConvNeXt-Tiny) as the 3-stage path, so capacity is comparable.
  * Trunk is ImageNet-pretrained but here trainable (the 3-stage path's trunk
    is frozen + pretrained on 2D pose; a single net has no 2D labels to freeze
    on, which is exactly R5's "one net" setting — trained end-to-end for 3D).
"""

import torch
import torch.nn as nn
from torchvision import models


class SingleNetSpatial(nn.Module):
    """Image -> per-joint tokens in one network (no heatmap, no tokenizer, no SJT)."""

    def __init__(self, num_joints=15, feature_dim=128, freeze_trunk=False):
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim

        backbone = models.convnext_tiny(weights='DEFAULT')
        self.trunk = backbone.features          # -> (B, 768, 8, 8) for 256x256 in
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)     # (B, 768, 1, 1)
        # One head regresses all J joint tokens jointly, then reshape.
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, images):
        """
        images: (B, T, 3, H, W)
        returns spatial_joint_features: (B, T, J, feature_dim)
        """
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)
        f = self.trunk(x)                       # (B*T, 768, 8, 8)
        f = self.pool(f).flatten(1)             # (B*T, 768)
        tok = self.head(f)                      # (B*T, J*dim)
        tok = tok.view(B * T, self.num_joints, self.feature_dim)
        tok = self.norm(tok)
        return tok.view(B, T, self.num_joints, self.feature_dim)


if __name__ == '__main__':
    m = SingleNetSpatial()
    x = torch.randn(2, 4, 3, 256, 256)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert y.shape == (2, 4, 15, 128), y.shape
    print(f"SingleNetSpatial out={tuple(y.shape)}  params={n/1e6:.2f}M  trainable={nt/1e6:.2f}M")
