import torch
import torch.nn as nn
from actionformer.modeling import make_meta_arch
from actionformer.config import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ActionFormerFeatureExtractor(nn.Module):
    def __init__(self, action_model):
        super(ActionFormerFeatureExtractor, self).__init__()
        self.model = action_model
        self.channel_projector = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=1)

    def forward(self, x, mask=None):
        if mask is None:
            x= x[:, :, :, 0, 0]# Shape: [16, 256, 3, 50176]
            print("Reshaped Image tensor:", x.shape)
            x = x.permute(0, 2, 1)
            x = self.channel_projector(x)
            print("Reshaped Image tensor:", x.shape)
            batch_size, channels, sequence_length = x.shape
            #Create a default mask: all elements are valid (True)
            mask = torch.ones((batch_size, channels, sequence_length), dtype=torch.bool, device=x.device)

        # mask: batch size, 1, sequence length (bool)
        features = {}

        # #Embedding layers
        # embd_out = []
        # for embd_layer in self.model.backbone.embd:
        #     x = embd_layer(x, mask)
        #     embd_out.append(x)
        # features['embd'] = embd_out
        #
        # #Stem layers (attention blocks)
        # stem_out = []
        # for stem_layer in self.model.backbone.stem:
        #     x = stem_layer(x, mask)
        #     stem_out.append(x)
        # features['stem'] = stem_out

        stem_out = []
        for stem_layer in self.model.backbone.stem:
            stem_features = stem_layer(x, mask)
            stem_out.append(stem_features)
        features['stem'] = stem_out
        print("Stem layer integrated",type(features))

        #Branch layers (transformer blocks)
        branch_out = []
        for branch_layer in self.model.backbone.branch:
            branch_features = branch_layer(x, mask)
            branch_out.append(branch_features)
        features['branch'] = branch_out
        print("Branch layer integrated",type(features))



        #Classification head (penultimate layer for global features)
        cls_out = []
        for head_layer in self.model.cls_head.head:
            head_features = head_layer(x, mask)
            cls_out.append(head_features)
        features['cls_head'] = cls_out
        # Optional: Final classification layer output
        features['final_output'] = self.model.cls_head.cls_head(x, mask)
        print("Stem, Branch and Head layers integrated")
        return features

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
    # Extract features without updating weights

    # Freeze ActionFormer model weights
    for param in actionformer_model.parameters():
        param.requires_grad = False

    return actionformer_feature_extractor


# # Display keys of extracted features for verification
# print("Extracted feature types:", extracted_features.keys())
