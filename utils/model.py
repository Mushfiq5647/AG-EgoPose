import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FeatureEncoder(nn.Module):
    PATCH_SIZE = 14  # DINOv2-ViT patch size

    def __init__(self, actionformer_model=None, embed_size=384, skip_temporal=False,
                 dino_feature='cls', joint_local='none'):
        super(FeatureEncoder, self).__init__()
        self.skip_temporal = skip_temporal
        # R2 W6 ablation: 'cls' = global CLS token (default), 'patch_mean' =
        # mean of patch tokens. Same 384-dim, different spatial pooling.
        # This controls what feeds ActionFormer (the *global* action stream).
        self.dino_feature = dino_feature
        # joint_local controls the *per-joint* stream L_{t,j}, which is
        # separate from the ActionFormer input: 'none' reproduces the BMVC
        # model, the other modes attach a per-joint DINO descriptor.
        self.joint_local = joint_local
        # DINOv2-ViT-S/14: self-supervised visual backbone (384-dim CLS token)
        # Frozen — provides domain-general features robust to fisheye distortion
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.eval()

        # DINOv2-ViT-S/14 outputs 384-dim, matching ActionFormer embed_dim directly
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.actionformer_model = actionformer_model

        # Pooling is done inside the per-frame loop so the (B, N, 384) patch
        # grid is never materialized for all T frames at once: at T=64, B=8,
        # N=324 that would be ~250 MB, versus ~12 MB for the pooled result.
        if joint_local != 'none':
            from utils.cross_attention_model import HeatmapGuidedPatchPooling
            self.patch_pool = HeatmapGuidedPatchPooling(dim=embed_size, mode=joint_local)

    def train(self, mode=True):
        super().train(mode)
        # DINOv2 must always stay in eval mode (frozen, no running stat updates)
        self.dinov2.eval()
        return self

    @staticmethod
    def _make_divisible(size, patch_size=14):
        """Round down to nearest multiple of patch_size."""
        return (size // patch_size) * patch_size

    def forward(self, images, heatmaps=None):
        """
        Args:
            images:   (B, T, 3, H, W)
            heatmaps: (B, T, J, Hh, Wh) — required when joint_local != 'none'
        Returns:
            motion_features: (B, T, 384) — ActionFormer output (or raw DINOv2
                             features when the temporal stream is skipped)
            joint_local:     (B, T, J, 384) per-joint DINO descriptors, or
                             None when joint_local == 'none'
        """
        B, T = images.shape[:2]
        want_local = self.joint_local != 'none'
        if want_local and heatmaps is None:
            raise ValueError(f"joint_local='{self.joint_local}' requires heatmaps")

        images_t = images.transpose(0, 1)  # (T, B, 3, H, W)
        feat_block = []
        local_block = []
        for t, batch in enumerate(images_t):  # batch: (B, 3, H, W)
            H, W = batch.shape[-2:]
            new_H = self._make_divisible(H)
            new_W = self._make_divisible(W)
            if new_H != H or new_W != W:
                batch = F.interpolate(batch, size=(new_H, new_W), mode='bilinear', align_corners=False)
            with torch.no_grad():
                if want_local:
                    # one forward pass serves both streams
                    out = self.dinov2.forward_features(batch)
                    patches = out['x_norm_patchtokens']          # (B, N, 384)
                    cls = out['x_norm_clstoken']                 # (B, 384)
                    features = patches.mean(dim=1) if self.dino_feature == 'patch_mean' else cls
                elif self.dino_feature == 'patch_mean':
                    # mean-pool patch tokens instead of the global CLS token
                    out = self.dinov2.forward_features(batch)
                    features = out['x_norm_patchtokens'].mean(dim=1)  # (B, 384)
                else:
                    features = self.dinov2(batch)  # (B, 384) — CLS token
            feat_block.append(features)
            if want_local:
                # pooling weights are learnable, so this stays outside no_grad
                local_block.append(self.patch_pool(patches, heatmaps[:, t], cls_token=cls))

        feat_block = torch.stack(feat_block, dim=1)  # (B, T, 384)
        joint_local = torch.stack(local_block, dim=1) if want_local else None  # (B,T,J,384)

        # Apply BN over (B*T, 384) to avoid single-sample error when B=1
        feat_flat = feat_block.reshape(B * T, -1)
        feat_flat = self.bn(feat_flat)
        feat_block = feat_flat.reshape(B, T, -1)

        if self.skip_temporal or self.actionformer_model is None:
            return feat_block, joint_local  # raw DINOv2 features (B, T, 384)

        actionformer_features = self.actionformer_model(feat_block)
        return actionformer_features, joint_local

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

class MLPPoseDecoder(nn.Module):
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

