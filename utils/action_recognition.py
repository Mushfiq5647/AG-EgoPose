import torch
import torch.nn as nn
from actionformer.modeling import make_meta_arch
from actionformer.config import load_config
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActionFormerFeatureExtractor(nn.Module):
    def __init__(self, action_model, dinov2_dim=384, actionformer_input_dim=256):
        super().__init__()
        self.model = action_model

        # Bridge DINOv2 (384-dim) to ActionFormer's expected input (256-dim)
        # The pretrained backbone embedding convs expect 256-dim input (EgoVLP features).
        # This learnable projection adapts DINOv2 features to the pretrained input space.
        self.input_proj = nn.Conv1d(dinov2_dim, actionformer_input_dim, kernel_size=1)

        # Number of multi-scale outputs from the backbone:
        # backbone.forward() returns (stem_output, branch_0, ..., branch_N)
        # That's 1 + len(branch) total feature maps
        n_scales = 1 + len(self.model.backbone.branch)
        embd_dim = self.model.backbone.branch[0].ln1.num_channels  # 384

        # Channel projector fuses all multi-scale features back to embd_dim
        self.channel_projector = nn.Conv1d(
            in_channels=embd_dim * n_scales,
            out_channels=embd_dim,
            kernel_size=1
        )

    def forward(self, x, mask=None):
        """
        x: (B, T, C) — DINOv2 per-frame features, C=384
        mask: (optional) boolean mask of shape (B, 1, T)
        returns: (B, T, embd_dim) fused temporal features
        """
        B, T, C = x.shape

        # (B, T, C) -> (B, C, T) for 1D conv / ActionFormer backbone
        x = x.permute(0, 2, 1)

        # Project DINOv2 384-dim -> ActionFormer's expected 256-dim input
        x = self.input_proj(x)  # (B, 256, T)

        if mask is None:
            mask = torch.ones((B, 1, T), dtype=torch.bool, device=x.device)

        # Run the FULL backbone: embedding convs -> stem transformers -> branch transformers
        # Returns tuple of multi-scale features + masks
        out_feats, out_masks = self.model.backbone(x, mask)
        # out_feats: (stem_out, branch_0_out, ..., branch_6_out) — all (B, embd_dim, T_i)

        # Upsample all scales back to original temporal length T and fuse
        upsampled = [
            F.interpolate(feat, size=T, mode='nearest')
            for feat in out_feats
        ]

        # Concatenate along channels -> (B, embd_dim * n_scales, T)
        cat = torch.cat(upsampled, dim=1)

        # Fuse back to embd_dim via 1x1 Conv1d
        fused = self.channel_projector(cat)  # (B, embd_dim, T)

        # Return time-major: (B, T, embd_dim)
        return fused.permute(0, 2, 1)


def initialize_actionformer(config_file_path, random_init=False):
    config = load_config(config_file_path)
    actionformer_model = make_meta_arch(config['model_name'], **config['model'])

    if random_init:
        print("ABLATION: ActionFormer randomly initialized (no pretrained weights)")
    else:
        checkpoint = torch.load(
            'actionformer/ego4d_egovlp_reproduce/epoch_010.pth.tar',
            map_location=torch.device('cuda')
        )
        state_dict = checkpoint['state_dict']
        new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        actionformer_model.load_state_dict(new_state_dict)

    actionformer_model = actionformer_model.to(device)
    actionformer_model.eval()

    # Wrap the model with the feature extractor
    actionformer_feature_extractor = ActionFormerFeatureExtractor(actionformer_model)
    actionformer_feature_extractor = actionformer_feature_extractor.to(device)

    # Freeze all parameters initially (selective unfreezing happens in train.py)
    for param in actionformer_model.parameters():
        param.requires_grad = False

    return actionformer_feature_extractor
