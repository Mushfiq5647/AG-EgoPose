import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FeatureEncoder(nn.Module):
    PATCH_SIZE = 14  # DINOv2-ViT patch size

    def __init__(self, actionformer_model, embed_size=384):
        super(FeatureEncoder, self).__init__()
        # DINOv2-ViT-S/14: self-supervised visual backbone (384-dim CLS token)
        # Frozen — provides domain-general features robust to fisheye distortion
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.eval()

        # DINOv2-ViT-S/14 outputs 384-dim, matching ActionFormer embed_dim directly
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # Symmetric stereo fusion on global CLS features.
        # mean captures shared scene/action context, |diff| preserves asymmetric
        # occlusion/viewpoint cues without doubling ActionFormer input width.
        self.stereo_fusion_proj = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size)
        )
        self.stereo_fusion_gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.Sigmoid()
        )
        self.actionformer_model = actionformer_model

    def train(self, mode=True):
        super().train(mode)
        # DINOv2 must always stay in eval mode (frozen, no running stat updates)
        self.dinov2.eval()
        return self

    @staticmethod
    def _make_divisible(size, patch_size=14):
        """Round down to nearest multiple of patch_size."""
        return (size // patch_size) * patch_size

    def _encode_view(self, images):
        """Run DINOv2 + BN frame-by-frame on a single view. Returns (B, T, 384)."""
        feat_block = []
        images = images.transpose(0, 1)  # (T, B, 3, H, W)
        for batch in images:  # batch: (B, 3, H, W)
            # Resize to nearest patch-aligned resolution (e.g. 256 → 252)
            H, W = batch.shape[-2:]
            new_H = self._make_divisible(H)
            new_W = self._make_divisible(W)
            if new_H != H or new_W != W:
                batch = F.interpolate(batch, size=(new_H, new_W), mode='bilinear', align_corners=False)
            with torch.no_grad():
                features = self.dinov2(batch)  # (B, 384) CLS token
            features = self.bn(features)
            feat_block.append(features)
        return torch.stack(feat_block, dim=1)  # (B, T, 384)

    def _fuse_stereo_views(self, feat_left, feat_right):
        """Lightweight symmetric stereo fusion on global CLS features."""
        mean_feat = 0.5 * (feat_left + feat_right)
        diff_feat = torch.abs(feat_left - feat_right)
        fusion_input = torch.cat([mean_feat, diff_feat], dim=-1)
        residual = self.stereo_fusion_proj(fusion_input)
        gate = self.stereo_fusion_gate(fusion_input)
        return mean_feat + gate * residual

    def forward(self, images_left, images_right=None):
        """
        Args:
            images_left:  (B, T, 3, H, W)
            images_right: (B, T, 3, H, W) or None for monocular
        Returns:
            actionformer_features: (B, T, 384)

        Stereo strategy: symmetric mean-plus-difference fusion before ActionFormer.
        DINOv2 produces one global CLS token per view; mean keeps shared action context,
        while |left-right| preserves asymmetric occlusion/viewpoint cues. A small gated
        residual projector fuses both without doubling ActionFormer input width.
        """
        feat_left = self._encode_view(images_left)
        if images_right is not None:
            feat_right = self._encode_view(images_right)
            feat_block = self._fuse_stereo_views(feat_left, feat_right)
        else:
            feat_block = feat_left

        actionformer_features = self.actionformer_model(feat_block)
        return actionformer_features

    # def sample(self, features, homography, openpose, states=None):
    #     sampled_ids = []
    #     embeddings = torch.zeros([1, 0]).to(features.device)  # Adjust size as needed
    #     device = features.device
    #     homography = homography.to(device)
    #     openpose = openpose.to(device)
    #     print("Feature shape", features.shape)
    #     print("Homography shape", homography.shape)
    #     features = features.squeeze(0)
    #     tensor = torch.cat((features, homography, openpose), dim=1)  # Now shape should be [1, 466]
    #     # Pass to GCN
    #     gcn_outputs = self.temporal_gcn(tensor)
    #     gcn_outputs = gcn_outputs.unsqueeze(0)
    #     print("GCN output", gcn_outputs.shape)
    #     hiddens, states = self.lstm(gcn_outputs, states)
    #     outputs = self.linear(hiddens[0])
    #     print("LSTM output", gcn_outputs.shape)
    #     return outputs

class PoseDecoder(nn.Module):
    def __init__(self, motion_dim=384, joint_dim=128, out_dim=3, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(motion_dim + joint_dim, hidden),
            nn.LeakyReLU(negative_slope=0.1),  # Better for negative values
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.LeakyReLU(negative_slope=0.1),  # Better for negative values
            nn.Dropout(0.1),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, enhanced_joints, motion_feats):
        # enhanced_joints: (B,T,J,128)
        # motion_feats:    (B,T,384)
        B,T,J,_ = enhanced_joints.shape
        # tile motion to joints
        motion_tiled = motion_feats.unsqueeze(2).expand(-1, -1, J, -1)   # (B,T,J,384)
        z = torch.cat([motion_tiled, enhanced_joints], dim=-1)           # (B,T,J,512)
        z = z.view(B*T*J, -1)                                            # (B*T*J, 512)
        y = self.mlp(z)                                                  # (B*T*J, out_dim)
        return y.view(B, T, J, -1)                                       # (B,T,J,out_dim)
