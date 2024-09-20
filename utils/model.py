# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# Enable CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class EncoderCNN(nn.Module):
	def __init__(self, embed_size, actionformer_model):
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
		self.actionformer_model = actionformer_model


	def forward(self, images):
		# images = images.transpose(0, 1)
		feat_block = []
		with torch.no_grad():
			actionformer_features = self.actionformer_model(images)
			# Assuming 'features' is the dictionary containing the keys 'cls_head' and 'final_output'
			stem_features = actionformer_features.get('stem')
			branch_features = actionformer_features.get('branch')
			# with open('action.txt', 'w') as f:
			# 	f.write(str(branch_features))
			cls_head = actionformer_features.get('cls_head')
			head = actionformer_features.get('final_output')
			# Print the keys where the values are tuples
			def conv_features(features):
				device = features.device
				features = features.permute(0, 2, 1)
				in_channel = features.size(1)
				conv1d_layer = nn.Conv1d(in_channel, out_channels=256, kernel_size=1).to(device)
				reduced_features = conv1d_layer(features)
				# Swap dimensions back to the original format [batch_size, channels, sequence_length]
				reduced_features = reduced_features.permute(0, 2, 1)
				return reduced_features


			def concat_features(features):
				if features is not None:
					processed_features = []

					# Iterate through each tuple in branch_features
					for idx, feature_tuple in enumerate(features):
						if isinstance(feature_tuple, tuple):
							# Concatenate the two tensors within each tuple along the last dimension (dim=2)
							concatenated_tuple_features = torch.cat(feature_tuple, dim=2)
							processed_features.append(concatenated_tuple_features)

					# Concatenate all processed tensors from each tuple along the last dimension (dim=2)
					final_concatenated_features = torch.cat(processed_features, dim=2)
					# print("Shape after final concatenation:", final_concatenated_features.shape)
					return final_concatenated_features

			# check_elements(head)
			concat_stem_features = concat_features(stem_features)
			# print("Shape of concatenated stem features:", concat_stem_features.shape)
			compact_stem_features = conv_features(concat_stem_features)
			# print("Shape of compact stem features", compact_stem_features.shape)

			concat_branch_features = concat_features(branch_features)
			# print("Shape of concatenated branch features:", concat_branch_features.shape)
			compact_branch_features = conv_features(concat_branch_features)
			# print("Shape of compact branch features", compact_branch_features.shape)



		images = images.transpose(0, 1)
		for batch in images:
			with torch.no_grad():
				features = self.resnet(batch)
			features = features.reshape(features.size(0), -1)
			features = self.bn(self.linear(features))
			feat_block.append(features)
		feat_block = torch.stack(feat_block, dim=1)
		combined_features = torch.cat((feat_block, compact_stem_features, compact_branch_features), dim=-1)
		print("Feature block", type(feat_block), combined_features.shape)
		return combined_features


class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, sequence_length, num_layers, use_homog=True, use_pose2=True, output_size=75,
				 num_homog=15, homog_size=9, pose2_size=75):
		super(DecoderRNN, self).__init__()
		# self.embed = nn.Embedding(vocab_size, embed_size)
		self.use_homog = use_homog
		self.use_pose2 = use_pose2
		self.embed_size = embed_size
		self.output_size = output_size
		self.lstm_full_input_size = output_size + sequence_length*3 +  (homog_size * num_homog) + pose2_size
		self.lstm_reduced_input_size = output_size + sequence_length*3
		if use_homog and use_pose2:
			self.lstm_input_size = self.lstm_full_input_size
		else:
			self.lstm_input_size = self.lstm_reduced_input_size
		self.input_projection = nn.Linear(self.lstm_reduced_input_size, self.lstm_full_input_size)
		self.lstm = nn.LSTM(self.lstm_full_input_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, (output_size))
		# self.embed_size = embed_size
		self.num_homog = num_homog
		self.homog_size = homog_size
		self.pose2_size = pose2_size
		self.output_size = output_size
		self.sequence_length = sequence_length

	def forward(self, features, gt_poses, lengths, homography=None, poses2=None):
		"""Decode image feature vectors and generate pose sequences."""
		device = features.device
		gt_poses = gt_poses.to(device)

		# Concatenate along the last dimension (gt_poses, features, homography, and poses2)
		embeddings = torch.cat((gt_poses, features), dim=-1)
		print("Embeddings shape", embeddings.shape)
		if self.use_homog and self.use_pose2:
			print(type(homography))
			homography = homography.to(device)
			poses2 = poses2.to(device)
			embeddings = torch.cat((embeddings, homography, poses2), dim=-1)
		else:
			embeddings = self.input_projection(embeddings)

		print("Embeddings shape:", embeddings.size(-1))
		print("LSTM size:", self.lstm_input_size)

		# assert embeddings.size(-1) == self.lstm_input_size, \
		# 	f"Input size mismatch: expected {self.lstm_input_size}, but got {embeddings.size(-1)}"


		# Pack the padded sequence
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

		# Pass through LSTM
		hiddens, _ = self.lstm(packed)

		# Pass the hidden states through the linear layer to get the final pose regression output
		outputs = self.linear(hiddens[0])
		print("Output shape", outputs.shape)# We are predicting 75 pose coordinates

		return outputs


	def sample(self, features, homography, openpose, states=None):
		sampled_ids = []
		embeddings = torch.zeros([1, 1, self.output_size]).cuda().float()
		features = features.squeeze(0)


		output_list = []
		for i in range(features.shape[0]):
			curr_feat = features[i].unsqueeze(0).unsqueeze(1)
			curr_h = homography[i].unsqueeze(0).unsqueeze(1)
			# print("Homography shape", curr_h.shape)
			curr_op = openpose[i].unsqueeze(0).unsqueeze(1)
			# print("Homography shape", curr_h.shape)
			tensor = torch.cat((embeddings, curr_feat), 2)
			tensor = torch.cat((tensor, curr_h.cuda()), 2)
			tensor = torch.cat((tensor, curr_op.cuda()), 2)	
			hiddens, states = self.lstm(tensor, states)
			outputs = self.linear(hiddens[0])
			output_list.append(outputs)
			all_outputs = torch.cat(output_list, dim=0)  # Shape: [256, output_size]
			# print("Output shape", outputs.shape)
		return all_outputs
		# 	_, predicted = outputs.max(1)
		# 	sampled_ids.append(predicted)
		# 	# embeddings = self.embed(predicted)
		# 	# embeddings = embeddings.unsqueeze(1)
		# sampled_ids = torch.stack(sampled_ids, 1)
		# return sampled_ids



