import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class JointIDEncoding(nn.Module):
    def __init__(self, num_joints=15, dim=128, p_drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_joints, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)  # tiny init
        self.alpha = nn.Parameter(torch.zeros(1))             # start from 0 => no effect
        self.dropout = nn.Dropout(p_drop)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B,T,J,dim)
        B,T,J,D = x.shape
        ids = torch.arange(J, device=x.device)
        pos = self.emb(ids).view(1,1,J,D)
        y = x + torch.sigmoid(self.alpha) * self.dropout(pos)
        return self.ln(y)


class HeatmapToJointFeatures(nn.Module):
    def __init__(self, heatmap_size=64, feature_dim=128, method='conv_pool'):
        super(HeatmapToJointFeatures, self).__init__()
        self.method = method
        self.feature_dim = feature_dim
        
        if method == 'adaptive_pool':
            # Simple adaptive pooling
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(1, feature_dim)
            
        # elif method == 'conv_pool':
        #     # Convolutional feature extraction
        #     self.conv_layers = nn.Sequential(
        #         nn.Conv2d(1, 32, 3, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.AdaptiveAvgPool2d((4, 4))
        #     )
        #     self.proj = nn.Linear(64 * 16, feature_dim)
        #     self.layer_norm = nn.LayerNorm(feature_dim)  # Normalize heatmap features

        elif method == 'conv_pool':
            # Balanced approach: Light conv + spatial pooling
            # GELU + BatchNorm prevent dead neurons (ReLU kills gradients on sigmoid heatmap inputs)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),       # 64x64 -> 64x64
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 64x64 -> 32x32
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32 -> 16x16
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4))           # 16x16 -> 4x4
            )
            self.proj = nn.Linear(64 * 4 * 4, feature_dim)  # 1024 -> 128
            self.layer_norm = nn.LayerNorm(feature_dim)


        elif method == 'spatial_attention':
            # Spatial attention pooling
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(heatmap_size * heatmap_size, feature_dim)
            
    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (B, T, J, H, W)
        Returns:
            joint_features: (B, T, J, feature_dim)
        """
        B, T, J, H, W = heatmaps.shape
        
        # Reshape for processing: (B*T*J, 1, H, W)
        heatmaps_flat = heatmaps.reshape(B*T*J, 1, H, W)
        
        if self.method == 'adaptive_pool':
            # Global average pooling
            pooled = self.pool(heatmaps_flat)  # (B*T*J, 1, 1, 1)
            pooled = pooled.squeeze(-1).squeeze(-1)  # (B*T*J, 1)
            features = self.proj(pooled)  # (B*T*J, feature_dim)
            
        elif self.method == 'conv_pool':
            # Balanced conv + spatial pooling
            conv_features = self.conv_layers(heatmaps_flat)  # (B*T*J, 64, 4, 4)
            conv_features = conv_features.view(B*T*J, -1)  # (B*T*J, 64*4*4)
            features = self.proj(conv_features)  # (B*T*J, feature_dim)
            features = self.layer_norm(features)  # Normalize feature scales
            
        elif self.method == 'spatial_attention':
            # Spatial attention weighted pooling
            attention_weights = self.spatial_attention(heatmaps_flat)  # (B*T*J, 1, H, W)
            weighted_heatmaps = heatmaps_flat * attention_weights  # (B*T*J, 1, H, W)
            pooled = weighted_heatmaps.view(B*T*J, -1)  # (B*T*J, H*W)
            features = self.proj(pooled)  # (B*T*J, feature_dim)
            
        # Reshape back to original structure
        joint_features = features.view(B, T, J, self.feature_dim)
        
        return joint_features


class SpatialStatsExtractor(nn.Module):
    """Lightweight differentiable stats from heatmaps for AIVE visibility gating.

    Computes 8 uncertainty descriptors per joint: (x, y, var_x, var_y,
    cov_xy, entropy, peak, top2_gap). These feed ONLY into the AIVE
    visibility gate — the main spatial feature path uses the conv encoder.

    No learnable parameters: purely functional stats extraction.
    """

    NUM_STATS = 8

    def __init__(self, heatmap_size=64, temperature=10.0):
        super().__init__()
        self.temperature = temperature

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, heatmap_size),
            torch.linspace(0, 1, heatmap_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.reshape(-1))
        self.register_buffer('grid_y', grid_y.reshape(-1))

    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (B, T, J, H, W)
        Returns:
            spatial_stats: (B, T, J, 8)
        """
        B, T, J, H, W = heatmaps.shape
        N = B * T * J
        hm = heatmaps.reshape(N, H * W)

        probs = F.softmax(hm * self.temperature, dim=-1)
        x = (probs * self.grid_x).sum(dim=-1)
        y = (probs * self.grid_y).sum(dim=-1)

        dx = self.grid_x.unsqueeze(0) - x.unsqueeze(1)
        dy = self.grid_y.unsqueeze(0) - y.unsqueeze(1)
        var_x = (probs * dx * dx).sum(dim=-1)
        var_y = (probs * dy * dy).sum(dim=-1)
        cov_xy = (probs * dx * dy).sum(dim=-1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        peak = hm.amax(dim=-1)

        top2 = hm.topk(2, dim=-1).values
        top2_gap = top2[:, 0] - top2[:, 1]

        stats = torch.stack([x, y, var_x, var_y, cov_xy, entropy, peak, top2_gap], dim=-1)
        return stats.view(B, T, J, self.NUM_STATS)


class StereoFeatureFusion(nn.Module):
    """Stereo fusion via gated concat-and-project with mean residual.

    Best practice for clean stereo features:
      - Mean baseline preserves view-invariant signal (works even if one view occluded).
      - Concatenate-and-project preserves per-view information that mean would average away
        (e.g. left wrist visible, right wrist occluded — mean discards this asymmetry).
      - Sigmoid gate per (B,T,J) decides how much projected disparity-aware signal to add
        on top of the mean.
      - LayerNorm stabilizes downstream training.

    Shared along the joint dim — same weights for all 15 joints (cheap, generalizes).
    Operates on per-joint features of shape (B, T, J, dim).
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Linear(dim * 2, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_left, f_right):
        """
        Args:
            f_left, f_right: (B, T, J, dim)
        Returns:
            fused: (B, T, J, dim)
        """
        cat = torch.cat([f_left, f_right], dim=-1)
        mean_f = 0.5 * (f_left + f_right)
        residual = self.proj(cat)
        g = torch.sigmoid(self.gate(cat))
        return self.norm(mean_f + g * residual)


class SpatialJointTransformer(nn.Module):
    """Spatial transformer for modeling joint relationships"""
    
    def __init__(self, feature_dim=128, num_heads=4, num_layers=3, dropout=0.1, num_joints=15):
        super(SpatialJointTransformer, self).__init__()
        self.feature_dim = feature_dim

        self.joint_id = JointIDEncoding(num_joints=num_joints, dim=feature_dim, p_drop=0.1)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, joint_features):
        """
        Args:
            joint_features: (B, T, J, feature_dim)
        Returns:
            enhanced_joints: (B, T, J, feature_dim)
        """
        B, T, J, feature_dim = joint_features.shape

        # Add learned joint identity embeddings (breaks permutation equivariance, resolves left/right ambiguity)
        joint_features = self.joint_id(joint_features)  # (B, T, J, feature_dim)

        joint_input = joint_features.view(B*T, J, feature_dim)
        enhanced_joints = self.transformer(joint_input)  # (B*T, J, feature_dim)
        
        # Layer normalization

        
        # Reshape back: (B, T, J, feature_dim)
        enhanced_joints = enhanced_joints.view(B, T, J, feature_dim)
        
        return enhanced_joints


class PerJointTrajectoryTokens(nn.Module):
    """Per-joint cross-attention on the ActionFormer temporal sequence
    with FiLM modulation for time-aware trajectory tokens.

    Each joint has a learnable query that independently attends to the
    temporal feature sequence, extracting joint-specific motion context.
    FiLM modulation adds frame-specific motion dynamics so that the
    trajectory token for the wrist at frame 10 differs from frame 50.
    """

    def __init__(self, num_joints=15, motion_dim=384, num_heads=4):
        super().__init__()
        self.motion_dim = motion_dim
        self.joint_queries = nn.Parameter(torch.randn(num_joints, motion_dim))
        nn.init.normal_(self.joint_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            motion_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(motion_dim)

        # Joint-aware FiLM modulation: per-frame motion + joint query → joint-specific offset
        # tau_{t,j} = tau_j + film_proj([m_t, tau_j])  — each joint gets distinct temporal modulation
        self.film_proj = nn.Sequential(
            nn.Linear(motion_dim * 2, motion_dim),
            nn.GELU(),
            nn.Linear(motion_dim, motion_dim),
            nn.LayerNorm(motion_dim)
        )

    def forward(self, motion_seq):
        """
        Args:
            motion_seq: (B, T, motion_dim) — ActionFormer temporal output
        Returns:
            traj_tokens: (B, T, J, motion_dim) — time-varying per-joint trajectory tokens
        """
        B, T, D = motion_seq.shape
        J = self.joint_queries.shape[0]

        # Joint-specific cross-attention over full temporal sequence
        queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)  # (B, J, D)
        traj_tokens, _ = self.cross_attn(queries, motion_seq, motion_seq)
        traj_tokens = self.layer_norm(traj_tokens + queries)  # (B, J, D) — joint identity tokens

        # Joint-aware FiLM: each joint gets a distinct temporal modulation
        # Combine per-frame motion (B,T,D) with per-joint tokens (B,J,D)
        motion_exp = motion_seq.unsqueeze(2).expand(-1, -1, J, -1)     # (B, T, J, D)
        joint_exp = traj_tokens.unsqueeze(1).expand(-1, T, -1, -1)     # (B, T, J, D)
        film_input = torch.cat([motion_exp, joint_exp], dim=-1)         # (B, T, J, 2D)
        film_offset = self.film_proj(film_input)                        # (B, T, J, D)
        # tau_{t,j} = tau_j + film([m_t, tau_j])
        traj_tokens = joint_exp + film_offset                           # (B, T, J, D)

        return traj_tokens


class ActionInformedVisibilityEstimation(nn.Module):
    """Cross-stream visibility gate using spatial uncertainty stats + trajectory tokens.

    Uses rich heatmap statistics (variance, entropy, peak, top2_gap) and
    time-varying trajectory tokens to decide per-joint, per-frame visibility.
    Visible joints trust spatial features; occluded joints fall back to
    motion-derived trajectory features.
    """

    def __init__(self, motion_dim=384, joint_dim=128, num_spatial_stats=8):
        super().__init__()
        # Gate now receives 8 spatial stats + motion_dim trajectory features
        self.visibility_gate = nn.Sequential(
            nn.Linear(num_spatial_stats + motion_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.traj_proj = nn.Linear(motion_dim, joint_dim)

    def forward(self, spatial_features, spatial_stats, traj_tokens):
        """
        Args:
            spatial_features: (B, T, J, joint_dim) — from SpatialJointTransformer
            spatial_stats: (B, T, J, 8) — rich uncertainty stats from SoftArgmaxJointEncoding
            traj_tokens: (B, T, J, motion_dim) — time-varying from PerJointTrajectoryTokens
        Returns:
            gated_features: (B, T, J, joint_dim) — visibility-gated joint features
        """
        B, T, J, D = spatial_features.shape

        # Gate input: [spatial_stats, tau_{t,j}]
        gate_input = torch.cat([
            spatial_stats,    # (B, T, J, 8) — x, y, var_x, var_y, cov_xy, entropy, peak, top2_gap
            traj_tokens       # (B, T, J, motion_dim) — time-varying trajectory tokens
        ], dim=-1)  # (B, T, J, 8 + motion_dim)

        v_j_raw = torch.sigmoid(
            self.visibility_gate(gate_input.reshape(B * T * J, -1))
        ).view(B, T, J, 1)

        # Clamp gate to [0.05, 0.95] to prevent sigmoid saturation from killing gradients
        v_j = 0.05 + 0.9 * v_j_raw

        # Project trajectory tokens to spatial feature dimension
        traj_feat = self.traj_proj(traj_tokens)  # (B, T, J, joint_dim)

        # Soft gate: visible → spatial, occluded → trajectory
        gated = v_j * spatial_features + (1.0 - v_j) * traj_feat
        return gated


class PoseDecoder(nn.Module):
    """Transformer decoder with joint queries for 3D pose prediction"""

    def __init__(self, joint_dim=128, motion_dim = 384, num_heads=4, num_layers=3, hidden=128, num_joints=15):
        super().__init__()
        self.num_joints = num_joints

        # Joint queries: learnable embeddings for each joint
        self.joint_queries = nn.Parameter(torch.randn(num_joints, joint_dim))
        nn.init.normal_(self.joint_queries, std=0.02)
        self.motion_proj = nn.Linear(motion_dim, joint_dim)
        # Project concatenated memory back to joint_dim
        self.memory_proj = nn.Sequential(
            nn.Linear(joint_dim*2, joint_dim),
            nn.ReLU(inplace=True),
            nn.Linear(joint_dim, joint_dim)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=joint_dim,
            nhead=num_heads,
            dim_feedforward=joint_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.pose_head = nn.Sequential(
            nn.Linear(joint_dim, hidden),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 3)  # Each joint → (x, y, z)
        )

    def forward(self, spatial_joint_features, motion_features):
        B, T, J, D = spatial_joint_features.shape

        # Joint queries attend to spatial features
        spatial_joint_features = spatial_joint_features.view(B * T, J, D)
        motion_features = self.motion_proj(motion_features)
        motion_tiled = motion_features.unsqueeze(2).expand(-1, -1, J, -1)  # (B,T,J,384)
        motion_tiled = motion_tiled.view(B * T, J, -1)
        # print("Motion tiled:", motion_tiled.shape)
        # print("Spatial tiled:", spatial_joint_features.shape)
        memory = torch.cat([spatial_joint_features,motion_tiled], dim=-1)
        # print("Memory shape", memory.shape)
        # Project memory back to joint_dim for transformer decoder
        memory = self.memory_proj(memory)  # (B*T, J, joint_dim)
        # # motion_memory = self.memory_proj(motion_tiled)  # (B*T, J, joint_dim)
        # # print("Motion memory tiled:", motion_memory.shape)
        queries = self.joint_queries.expand(B * T, self.num_joints, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        poses_3d = self.pose_head(decoded)

        return poses_3d.view(B, T, self.num_joints, 3)


class MotionPoseCrossAttention(nn.Module):
    """Cross attention between motion features and pose features"""

    def __init__(self, motion_dim=384, pose_dim=128, num_heads=8, dropout=0.1):
        super(MotionPoseCrossAttention, self).__init__()
        self.motion_dim = motion_dim
        self.pose_dim = pose_dim

        # Ensure dimensions are compatible
        self.hidden_dim = min(motion_dim, pose_dim)

        # Projection layers
        self.motion_to_query = nn.Linear(motion_dim, self.hidden_dim)
        self.pose_to_key = nn.Linear(pose_dim, self.hidden_dim)
        self.pose_to_value = nn.Linear(pose_dim, self.hidden_dim)

        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, motion_dim)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(motion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, motion_features, pose_features):
        """
        Args:
            motion_features: (B, T, motion_dim) - from ActionFormer
            pose_features: (B, T, J, pose_dim) - from joint transformer
        Returns:
            attended_motion: (B, T, motion_dim)
            attention_weights: (B, T, J) - which joints each timestep attends to
        """
        B, T, motion_dim = motion_features.shape
        B, T, J, pose_dim = pose_features.shape

        # Reshape pose features: (B, T*J, pose_dim)
        pose_flat = pose_features.view(B, T*J, pose_dim)

        # Project features
        query = self.motion_to_query(motion_features)  # (B, T, hidden_dim)
        key = self.pose_to_key(pose_flat)  # (B, T*J, hidden_dim)
        value = self.pose_to_value(pose_flat)  # (B, T*J, hidden_dim)

        # Cross attention: motion queries pose
        attended, attention_weights = self.cross_attention(
            query, key, value, need_weights=True
        )  # attended: (B, T, hidden_dim), weights: (B, T, T*J)

        # Project back to motion dimension
        attended_motion = self.output_proj(attended)  # (B, T, motion_dim)

        # Residual connection and normalization
        attended_motion = self.layer_norm(attended_motion + motion_features)
        attended_motion = self.dropout(attended_motion)

        # Reshape attention weights to per-joint: (B, T, J)
        attention_weights = attention_weights.view(B, T, T, J).mean(dim=2)  # Average across time

        return attended_motion, attention_weights


class JointAggregator(nn.Module):
    """Aggregate joint features for final fusion"""

    def __init__(self, joint_dim=128, num_joints=15, output_dim=128, method='attention'):
        super(JointAggregator, self).__init__()
        self.method = method
        self.num_joints = num_joints
        self.output_dim = output_dim

        if method == 'mean':
            # Simple mean pooling
            self.proj = nn.Linear(joint_dim, output_dim)

        elif method == 'attention':
            # Attention-based aggregation
            self.attention = nn.Sequential(
                nn.Linear(joint_dim, joint_dim // 2),
                nn.ReLU(),
                nn.Linear(joint_dim // 2, 1)
            )
            self.proj = nn.Linear(joint_dim, output_dim)

        elif method == 'flatten':
            # Flatten all joints
            self.proj = nn.Linear(joint_dim * num_joints, output_dim)

    def forward(self, joint_features):
        """
        Args:
            joint_features: (B, T, J, joint_dim)
        Returns:
            aggregated: (B, T, output_dim)
        """
        B, T, J, joint_dim = joint_features.shape

        if self.method == 'mean':
            # Mean across joints
            aggregated = joint_features.mean(dim=2)  # (B, T, joint_dim)
            aggregated = self.proj(aggregated)  # (B, T, output_dim)

        elif self.method == 'attention':
            # Attention-weighted aggregation
            attention_weights = self.attention(joint_features)  # (B, T, J, 1)
            attention_weights = F.softmax(attention_weights.squeeze(-1), dim=-1)  # (B, T, J)

            # Weighted sum
            weighted_joints = joint_features * attention_weights.unsqueeze(-1)  # (B, T, J, joint_dim)
            aggregated = weighted_joints.sum(dim=2)  # (B, T, joint_dim)
            aggregated = self.proj(aggregated)  # (B, T, output_dim)

        elif self.method == 'flatten':
            # Flatten all joints
            flattened = joint_features.view(B, T, -1)  # (B, T, J*joint_dim)
            aggregated = self.proj(flattened)  # (B, T, output_dim)

        return aggregated


class DualStreamPoseEstimator(nn.Module):
    """Complete dual-stream pose estimation model with cross attention"""
    def __init__(self,
                 actionformer_model,
                 motion_dim=384,
                 joint_feature_dim=128,
                 num_joints=15,
                 num_heads=4,
                 num_transformer_layers=3,
                 heatmap_size=128,
                 output_pose_dim=3):
        super(DualStreamPoseEstimator, self).__init__()

        self.actionformer_model = actionformer_model
        self.motion_dim = motion_dim
        self.joint_feature_dim = joint_feature_dim
        self.num_joints = num_joints

        # Heatmap to joint features conversion
        self.heatmap_converter = HeatmapToJointFeatures(
            heatmap_size=heatmap_size,
            feature_dim=joint_feature_dim,
            method='conv_pool'  # Best method for rich features
        )

        # Spatial joint transformer
        self.joint_transformer = SpatialJointTransformer(
            feature_dim=joint_feature_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers
        )

        # Cross attention fusion
        self.cross_attention = MotionPoseCrossAttention(
            motion_dim=motion_dim,
            pose_dim=joint_feature_dim,
            num_heads=num_heads
        )

        # Joint aggregation
        self.joint_aggregator = JointAggregator(
            joint_dim=joint_feature_dim,
            num_joints=num_joints,
            output_dim=128,
            method='attention'
        )

        # Final pose decoder
        self.pose_decoder = nn.Sequential(
            nn.Linear(motion_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_joints * output_pose_dim)
        )

        # Layer normalization
        self.final_norm = nn.LayerNorm(num_joints * output_pose_dim)

    def forward(self, image_features, heatmaps):
        """
        Args:
            image_features: (B, T, feature_dim) - preprocessed image features for ActionFormer
            heatmaps: (B, T, J, H, W) - predicted or GT heatmaps
        Returns:
            poses: (B, T, J, 3) - 3D pose predictions
            attention_weights: (B, T, J) - cross attention weights
            joint_features: (B, T, J, joint_dim) - enhanced joint features
        """
        B, T = image_features.shape[:2]

        motion_features = self.actionformer_model(image_features)  # (B, T, motion_dim)

        # Pose stream: Heatmaps to joint features
        joint_features = self.heatmap_converter(heatmaps)  # (B, T, J, joint_dim)

        # Spatial joint relationships
        enhanced_joints = self.joint_transformer(joint_features)  # (B, T, J, joint_dim)

        # Cross attention: Motion attends to pose
        attended_motion, attention_weights = self.cross_attention(
            motion_features, enhanced_joints
        )  # (B, T, motion_dim), (B, T, J)

        # Aggregate joint features
        aggregated_joints = self.joint_aggregator(enhanced_joints)  # (B, T, 128)

        # Fusion
        fused_features = torch.cat([attended_motion, aggregated_joints], dim=-1)  # (B, T, motion_dim+128)

        # Final pose prediction
        pose_logits = self.pose_decoder(fused_features)  # (B, T, J*3)
        pose_logits = self.final_norm(pose_logits)

        # Reshape to pose format
        poses = pose_logits.view(B, T, self.num_joints, 3)  # (B, T, J, 3)

        return poses, attention_weights, enhanced_joints

    def get_attention_visualization(self, attention_weights, joint_names=None):
        """
        Visualize which joints the model attends to for each timestep

        Args:
            attention_weights: (B, T, J) - attention weights from forward pass
            joint_names: List of joint names for visualization

        Returns:
            attention_summary: Dictionary with attention statistics
        """
        if joint_names is None:
            joint_names = [f"Joint_{i}" for i in range(self.num_joints)]

        # Average attention across batch and time
        avg_attention = attention_weights.mean(dim=[0, 1])  # (J,)

        # Top attended joints
        top_k = 5
        top_indices = torch.topk(avg_attention, top_k).indices
        top_joints = [joint_names[i] for i in top_indices]
        top_weights = avg_attention[top_indices]

        attention_summary = {
            'average_attention_per_joint': dict(zip(joint_names, avg_attention.tolist())),
            'top_attended_joints': dict(zip(top_joints, top_weights.tolist())),
            'attention_variance': attention_weights.var(dim=[0, 1]).tolist()
        }

        return attention_summary


def create_dual_stream_model(actionformer_model, config=None):
    """
    Factory function to create the dual stream model with default or custom config

    Args:
        actionformer_model: Pre-trained ActionFormer model
        config: Configuration dictionary (optional)

    Returns:
        model: DualStreamPoseEstimator instance
    """
    default_config = {
        'motion_dim': 384,
        'joint_feature_dim': 128,
        'num_joints': 15,
        'num_heads': 8,
        'num_transformer_layers': 3,
        'heatmap_size': 128,
        'output_pose_dim': 3
    }

    if config:
        default_config.update(config)

    model = DualStreamPoseEstimator(
        actionformer_model=actionformer_model,
        **default_config
    )

    return model


# Example usage and testing
if __name__ == "__main__":
    # Test the model components
    batch_size, seq_len, num_joints, heatmap_size = 2, 8, 15, 128
    motion_dim, joint_dim = 384, 128

    # Create dummy data
    image_features = torch.randn(batch_size, seq_len, motion_dim)
    heatmaps = torch.randn(batch_size, seq_len, num_joints, heatmap_size, heatmap_size)

    # Test heatmap converter
    print("Testing HeatmapToJointFeatures...")
    heatmap_converter = HeatmapToJointFeatures(heatmap_size, joint_dim, 'conv_pool')
    joint_features = heatmap_converter(heatmaps)
    print(f"Heatmaps {heatmaps.shape} -> Joint features {joint_features.shape}")

    # Test spatial transformer
    print("\nTesting SpatialJointTransformer...")
    joint_transformer = SpatialJointTransformer(joint_dim)
    enhanced_joints = joint_transformer(joint_features)
    print(f"Joint features {joint_features.shape} -> Enhanced joints {enhanced_joints.shape}")

    # Test cross attention
    print("\nTesting MotionPoseCrossAttention...")
    cross_attention = MotionPoseCrossAttention(motion_dim, joint_dim)
    attended_motion, attention_weights = cross_attention(image_features, enhanced_joints)
    print(f"Motion {image_features.shape} + Joints {enhanced_joints.shape} -> Attended {attended_motion.shape}")
    print(f"Attention weights: {attention_weights.shape}")

    # Test joint aggregator
    print("\nTesting JointAggregator...")
    aggregator = JointAggregator(joint_dim, num_joints, 128, 'attention')
    aggregated = aggregator(enhanced_joints)
    print(f"Enhanced joints {enhanced_joints.shape} -> Aggregated {aggregated.shape}")

    print("\nAll components tested successfully!")
