import torch
import torch.nn as nn
from actionformer.modeling import make_meta_arch
from actionformer.config import load_config
import torch.nn.functional as F
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActionFormerFeatureExtractor(nn.Module):
    def __init__(self, action_model):
        super().__init__()
        self.model = action_model

        # Spatial backbone (ResNet18 w/o avgpool+fc)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.spatial_backbone = nn.Sequential(*modules)
        self.spatial_proj = nn.Conv2d(512, 384, kernel_size=1)

        # We'll have len(self.model.backbone.branch) scales to fuse
        n_scales = len(self.model.backbone.branch) - 1
        print('n_scales', n_scales)
        # Channel-projector now needs in_channels = 256 * n_scales
        self.channel_projector = nn.Conv1d(
            in_channels=384 * n_scales,
            out_channels=384,
            kernel_size=1
        )

    def forward(self, x, mask=None):
        """
        x: (B, T, C)  — your precomputed per‑frame features, here C=384
        mask: (optional) boolean mask of shape (B, C, T)
        returns: (B, T, C) fused features
        """
        B, T, C = x.shape
        print("X shape", x.shape)
        # 1) prepare for temporal blocks: (B, C, T)
        x = x.permute(0, 2, 1)
        if mask is None:
            # mask must have the same channel dim as x
            mask = torch.ones((B, C, T), dtype=torch.bool, device=x.device)

        # 2) apply each block if its sliding window fits
        branch_scales = []
        for blk in self.model.backbone.branch:
            win = getattr(blk.attn, 'window_overlap', 1)
            seq_len = x.size(-1)

            # skip blocks whose window logic can’t handle this seq_len
            if seq_len < 2 * win or (seq_len % (2 * win) != 0):
                continue

            x, mask = blk(x, mask)
            branch_scales.append(x)

        # 3) upsample each scale back to length T
        upsampled = [
            F.interpolate(s, size=T, mode='nearest')  # each s: (B, C, smaller_T)
            for s in branch_scales
        ]

        # 4) concatenate along channels → (B, C * n_scales, T)
        cat = torch.cat(upsampled, dim=1)
        print("cat", cat.shape)

        # 5) fuse back down to C channels via your 1×1 Conv1d
        fused = self.channel_projector(cat)  # (B, C, T)

        # 6) return time‑first → (B, T, C)
        return fused.permute(0, 2, 1)


def initialize_actionformer(config_file_path):
    config = load_config(config_file_path)
    print(config['model_name'])
    actionformer_model = make_meta_arch(config['model_name'], **config['model'])
    checkpoint = torch.load('actionformer/ego4d_egovlp_reproduce/epoch_010.pth.tar', map_location=torch.device('cuda'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    actionformer_model.load_state_dict(new_state_dict)
    actionformer_model = actionformer_model.to(device)
    actionformer_model.eval()

    # Wrap the model with the feature extractor
    actionformer_feature_extractor = ActionFormerFeatureExtractor(actionformer_model)
    print(actionformer_feature_extractor)
    for param in actionformer_model.parameters():
        param.requires_grad = False
    actionformer_feature_extractor.eval()
    print("ActionFormerFeatureExtractor frozen and set to eval mode")

    return actionformer_feature_extractor
