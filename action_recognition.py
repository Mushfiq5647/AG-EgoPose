import torch
import torch.nn as nn

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

# # Display keys of extracted features for verification
# print("Extracted feature types:", extracted_features.keys())
