import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models

class FeatureEncoder(nn.Module):
    def __init__(self, actionformer_model, embed_size=384):
        super(FeatureEncoder, self).__init__()
        # Use ResNet-50 instead of ResNet-152 (better for egocentric vision)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for i, layer in enumerate(self.resnet):
            if i < 7:  # Freeze indices 0-6: conv1, bn1, relu, maxpool, layer1, layer2, layer3
                layer.eval()
                for p in layer.parameters():
                    p.requires_grad = False
            else:
                layer.train()
            # Index 7 (layer4) and Index 8 (avgpool) stay trainable
        
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.actionformer_model = actionformer_model
        
        # Set ActionFormer to eval mode if it's frozen
        if not any(p.requires_grad for p in actionformer_model.parameters()):
            self.actionformer_model.eval()
            print("Actionformer model set to eval")

    def forward(self, images):
        feat_block = []
        images = images.transpose(0, 1)
        for batch in images:
            # ResNet forward pass (last layer is trainable)
            features = self.resnet(batch)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))
            feat_block.append(features)
        
        feat_block = torch.stack(feat_block, dim=1)
        
        # ActionFormer parameters are frozen (requires_grad=False) but gradients can flow through
        actionformer_features = self.actionformer_model(feat_block)
        print("Actionformer features min/max:", actionformer_features.min().item(),
              actionformer_features.max().item())
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

