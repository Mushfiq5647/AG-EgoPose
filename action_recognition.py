import torch
import torch.nn as nn
from actionformer.modeling import make_meta_arch
from actionformer.config import load_config
import torch.nn.functional as F
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class ActionFormerFeatureExtractor(nn.Module):
#     def __init__(self, action_model):
#         super(ActionFormerFeatureExtractor, self).__init__()
#         self.model = action_model
#         self.channel_projector = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=1)
#
#     def forward(self, x, mask=None):
#         if mask is None:
#             x= x[:, :, :, 0, 0]# Shape: [16, 256, 3, 50176]
#             print("Reshaped Image tensor:", x.shape)
#             x = x.permute(0, 2, 1)
#             x = self.channel_projector(x)
#             print("Reshaped Image tensor:", x.shape)
#             batch_size, channels, sequence_length = x.shape
#             #Create a default mask: all elements are valid (True)
#             mask = torch.ones((batch_size, channels, sequence_length), dtype=torch.bool, device=x.device)
#         features = {}
#
#         stem_out = []
#         for stem_layer in self.model.backbone.stem:
#             stem_features = stem_layer(x, mask)
#             stem_out.append(stem_features)
#         features['stem'] = stem_out
#         print("Stem layer integrated",type(features))
#
#         #Branch layers (transformer blocks)
#         branch_out = []
#         for branch_layer in self.model.backbone.branch:
#             branch_features = branch_layer(x, mask)
#             branch_out.append(branch_features)
#         features['branch'] = branch_out
#         print("Branch layer integrated",type(features))
#         return features

class ActionFormerFeatureExtractor(nn.Module):
    def __init__(self, action_model):
        super().__init__()
        self.model = action_model

        # Spatial backbone (ResNet18 w/o avgpool+fc)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.spatial_backbone = nn.Sequential(*modules)
        self.spatial_proj = nn.Conv2d(512, 256, kernel_size=1)

        # We'll have len(self.model.backbone.branch) scales to fuse
        n_scales = len(self.model.backbone.branch)
        print('n_scales', n_scales)
        # Channel-projector now needs in_channels = 256 * n_scales
        self.channel_projector = nn.Conv1d(
            in_channels=256 * n_scales,
            out_channels=256,
            kernel_size=1
        )

    def forward(self, x, mask=None):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        # 1) Spatial feature extraction per frame
        x = x.view(B * T, C, H, W)                        # (B*T,3,H,W)
        feat = self.spatial_backbone(x)                   # (B*T,512,H',W')
        feat = self.spatial_proj(feat)                    # (B*T,256,H',W')
        feat = feat.view(B, T, 256, -1).mean(-1)          # (B,T,256)

        # 2) Prepare for temporal blocks: (B,256,T)
        x = feat.permute(0, 2, 1)
        if mask is None:
            mask = torch.ones((B, 256, T), dtype=torch.bool, device=x.device)

        # 4) Run through branch blocks sequentially, collecting each output
        branch_scales = []
        for blk in self.model.backbone.branch:
            x, mask = blk(x, mask)    # x: (B,256,L_i)
            branch_scales.append(x)

        # 5) Upsample each scale back to length T
        upsampled = [
            F.interpolate(s, size=T, mode='nearest')  # → (B,256,T)
            for s in branch_scales
        ]

        # 6) Concatenate along channels: (B, 256*n_scales, T)
        cat_channel = torch.cat(upsampled, dim=1)

        # 7) Fuse back down to 256 channels, still length T
        fused_feature = self.channel_projector(cat_channel)           # (B,256,T)
        fused_feature = fused_feature.permute(0, 2, 1)
        print("fused_feature.shape:", fused_feature.shape)
        # 8) Return time-first: (B, T, 256)
        return fused_feature


def initialize_actionformer(config_file_path):
    config = load_config(config_file_path)
    print(config['model_name'])
    actionformer_model = make_meta_arch(config['model_name'], **config['model'])
    checkpoint = torch.load('actionformer/epoch_015.pth.tar', map_location=torch.device('cuda'))
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

    return actionformer_feature_extractor
