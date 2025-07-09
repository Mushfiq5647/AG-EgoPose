import os
import torch
import torch.nn as nn
import torchvision.models as models

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, embed_feature_dim, num_layers, num_joints=16, feature_dim=256, num_heads=8, seq_len=256):
        super(SpatioTemporalTransformer, self).__init__()
        self.embed_dim = embed_feature_dim
        self.feature_dim = feature_dim
        self.num_joints = num_joints
        self.feature_embedding = nn.Linear(feature_dim, embed_feature_dim)
        self.embed_norm = nn.LayerNorm(embed_feature_dim)
        self.temporal_norm = nn.LayerNorm(embed_feature_dim)
        self.spatial_norm = nn.LayerNorm(embed_feature_dim)
        self.final_norm = nn.LayerNorm(embed_feature_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.0,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_feature_dim, self.num_joints * embed_feature_dim)
        self.joint_embed = nn.Linear(3, embed_feature_dim)
        spat_layer = nn.TransformerEncoderLayer(
            d_model=embed_feature_dim, nhead=2, dim_feedforward=128,
            dropout=0.0,
            activation='gelu'
        )
        self.spatial_refiner = nn.TransformerEncoder(spat_layer, num_layers=1)
        self.joint_proj = nn.Linear(embed_feature_dim, 3)

    def forward(self, x):
        B, T, F = x.shape
        # Feature Embedding
        print("Feature shape:", x.shape)
        # A) Check and normalize raw x
        print(f"[0] raw x range = ({x.min().item():.4f}, {x.max().item():.4f})")
        assert not torch.isnan(x).any(), "NaN in raw input!"
        emb = self.feature_embedding(x)
        print(f"[A] emb before norm = ({emb.min().item():.4f}, {emb.max().item():.4f})")
        assert not torch.isnan(emb).any(), "NaN after feature_embedding!"
        # emb = self.embed_norm(emb)
        print(f"[A'] emb after norm = ({emb.min().item():.4f}, {emb.max().item():.4f})")
        emb = emb.permute(1, 0, 2)
        t_out = self.temporal_encoder(emb)  # Temporal attention
        t_out = t_out.permute(1, 0, 2)
        # t_out = self.temporal_norm(t_out)
        pred = self.output_layer(t_out)
        pred = pred.view(B, T, self.num_joints, self.embed_dim)
        print("Pred min/max", pred.min().item(), pred.max().item())
        print("Pred shape", pred.shape)

        #Spatial Refinement
        pred_frames = pred.view(B * T, self.num_joints, self.embed_dim)  # (B*T, J, 3)
        # joint_emb = self.joint_embed(pred_frames)  # (B*T, J, D)
        # Transformer expects (seq_len, batch, embed)
        jr = pred_frames.permute(1, 0, 2)  # (J, B*T, D)
        jr = self.spatial_refiner(jr)  # (J, B*T, D)
        assert not torch.isnan(jr).any(), "NaN inside spatial_refiner"
        assert not torch.isinf(jr).any(), "Inf inside spatial_refiner"
        jr = jr.permute(1, 0, 2)  # (B*T, J, D)
        print("Joint shape", jr.shape)
        # t_out = self.spatial_norm(jr)
        # 5) Project back to 3D coords
        final_coords = self.joint_proj(jr)  # (B*T, J, 3)
        print("Input shape after joint projection", final_coords.shape)
        final_coords = final_coords.view(B, T, -1, 3)
        print("Final shape:", final_coords.shape)
        print("Final min/max:", final_coords.min().item(), final_coords.max().item())
        return pred, final_coords


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, actionformer_model):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.eval()
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.actionformer_model = actionformer_model

    def forward(self, images):
        # images = images.transpose(0, 1)
        feat_block = []
        with torch.no_grad():
            actionformer_features = self.actionformer_model(images)
            print("Actionformer features min/max:", actionformer_features.min().item(),
                  actionformer_features.max().item())

            # Assuming 'features' is the dictionary containing the keys 'cls_head' and 'final_output'
            # stem_features = actionformer_features.get('stem')
            # branch_features = actionformer_features.get('branch')

            def slice_window(features):
                device = features.device
                B, C, L_total = features.shape
                T = images.size(1)

                # permute to (B, L_total, C)
                feat = features.permute(0, 2, 1)

                # slice/pad
                if L_total >= T:
                    feat = feat[:, :T, :]
                else:
                    pad_len = T - L_total
                    padding = torch.zeros(B, pad_len, C, device=feat.device)
                    feat = torch.cat([feat, padding], dim=1)

                # feat is now (B, T, C)
                print("Sliced Feature shape", feat.shape)
                return feat

        # def concat_features(features):
        # 	if features is not None:
        # 		processed_features = []
        #
        # 		# Iterate through each tuple in branch_features
        # 		for idx, feature_tuple in enumerate(features):
        # 			if isinstance(feature_tuple, tuple):
        # 				# Concatenate the two tensors within each tuple along the last dimension (dim=2)
        # 				concatenated_tuple_features = torch.cat(feature_tuple, dim=2)
        # 				processed_features.append(concatenated_tuple_features)
        #
        # 		# Concatenate all processed tensors from each tuple along the last dimension (dim=2)
        # 		final_concatenated_features = torch.cat(processed_features, dim=2)
        # 		# print("Shape after final concatenation:", final_concatenated_features.shape)
        # 		return final_concatenated_features
        #
        # # concat_stem_features = concat_features(stem_features)
        # # compact_stem_features = conv_features(concat_stem_features)
        #
        # concat_branch_features = concat_features(branch_features)
        # print("concat_branch_features:", concat_branch_features.shape)
        # compact_branch_features = slice_window(concat_branch_features)
        # print("Branch layer shape", compact_branch_features.shape)

        images = images.transpose(0, 1)
        for batch in images:
            with torch.no_grad():
                features = self.resnet(batch)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))
            feat_block.append(features)
        feat_block = torch.stack(feat_block, dim=1)
        print("ResNet feat range:", feat_block.min().item(), feat_block.max().item())
        combined_features = torch.cat((feat_block, actionformer_features), dim=-1)
        print("Feature block", type(feat_block), combined_features.shape)
        return actionformer_features


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, sequence_length, spatio_temporal_transformer, use_homog=True,
                 use_pose2=True, output_size=48, num_homog=15, homog_size=9, pose2_size=48):
        super(TransformerDecoder, self).__init__()
        # self.embed = nn.Embedding(vocab_size, embed_size)
        self.use_homog = use_homog
        self.use_pose2 = use_pose2
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, (output_size))
        # self.embed_size = embed_size
        self.num_homog = num_homog
        self.homog_size = homog_size
        self.pose2_size = pose2_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.spatio_temporal_transformer = spatio_temporal_transformer

    def forward(self, features, lengths, homography=None, poses2=None):
        """Decode image feature vectors and generate pose sequences."""
        device = features.device
        if self.use_homog and self.use_pose2:
            print(type(homography))
            homography = homography.to(device)
            B, T, H, W = homography.shape  # e.g., (16, 32, 3, 3)
            homography = homography.view(B, T, H * W)  #
            homography = homography
            print(f"Homography min/max= ({homography.min().item():.4f}, {homography.max().item():.4f})")
            print("Shape of embedded features:", features.shape)
            print("Shape of homography features:", homography.shape)
            embeddings = torch.cat((features, homography), dim=-1)
            print("Embeddings shape", embeddings.shape)
        pred, final = self.spatio_temporal_transformer(features)
        # print("Output shape", transformer_output.shape)

        return pred, final

    def sample(self, features, homography, openpose, states=None):
        sampled_ids = []
        embeddings = torch.zeros([1, 0]).to(features.device)  # Adjust size as needed
        device = features.device
        homography = homography.to(device)
        openpose = openpose.to(device)
        print("Feature shape", features.shape)
        print("Homography shape", homography.shape)
        features = features.squeeze(0)
        output_list = []
        # for i in range(features.shape[0]):
        # 	curr_feat = features[i]  # Shape [1, feature_dim]
        # 	curr_h = homography[i]
        # 	curr_op = openpose[i]
        # 	print("Feature shape", curr_feat.shape)
        # 	print("Homography shape", curr_h.shape)

        tensor = torch.cat((features, homography, openpose), dim=1)  # Now shape should be [1, 466]
        # Pass to GCN
        gcn_outputs = self.temporal_gcn(tensor)
        gcn_outputs = gcn_outputs.unsqueeze(0)
        print("GCN output", gcn_outputs.shape)
        hiddens, states = self.lstm(gcn_outputs, states)
        outputs = self.linear(hiddens[0])
        print("LSTM output", gcn_outputs.shape)
        return outputs
