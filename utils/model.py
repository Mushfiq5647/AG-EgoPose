# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# Enable CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class EncoderCNN(nn.Module):
	def __init__(self, embed_size):
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


	def forward(self, images):
		images = images.transpose(0, 1)
		feat_block = []
		for batch in images:
			with torch.no_grad():
				features = self.resnet(batch)
			features = features.reshape(features.size(0), -1)
			features = self.bn(self.linear(features))
			feat_block.append(features)
		feat_block = torch.stack(feat_block, dim=1)
		return feat_block


class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, output_size, num_layers,
				 num_homog=15, homog_size=9, pose2_size=48):
		super(DecoderRNN, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM((embed_size*3) + (homog_size * num_homog) + pose2_size,
			hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, (output_size+1))
		print("Low Body Pose:", output_size)
		self.embed_size = embed_size
		self.num_homog = num_homog
		self.homog_size = homog_size
		self.pose2_size = pose2_size


	def forward(self, features, poses, homography, poses2, lengths):
		"""Decode image feature vectors and generate pose sequences."""
		device = features.device

		# Embed poses and move to the correct device
		print(f"Poses shape before embedding: {poses.shape}")
		with open('sample_poses.txt', 'a') as f:
			f.write(f'{poses[0]}\n')
		# print(f"Poses before embedding: {poses[0]}")
		print(f"Poses min/max values: {poses.min()}, {poses.max()}")
		print("Embedding layer input size:", self.embed.num_embeddings)
		# poses = self.embed(poses).to(device)

		try:
			poses = self.embed(poses).to(device)
		except Exception as e:
			print(f"Error in embedding: {e}")

			print("Poses:", poses)
			raise

		# Print shapes for debugging
		# print(f"Embedded poses shape: {poses.shape}")
		# print(f"Features shape: {features.shape}")
		# print(f"Homography shape: {homography.shape}")
		# print(f"Poses2 shape: {poses2.shape}")
		# Flatten the poses to match the dimensions of features, homography, and poses2
		# This changes poses from [32, 512, 2, 256] to [32, 512, 512]
		poses = poses.view(poses.size(0), poses.size(1), -1)
		print(f"Reshaped poses shape: {poses.shape}")
		poses[:, 0, :] = 0
		# Move other tensors to the correct device
		print(device)
		homography = homography.to(device)
		print(f"Homography shape: {homography.shape}")
		print(f"Poses shape: {poses.shape}")

		print(f"Any NaNs in homography: {torch.isnan(homography).any()}")
		print(f"Any infinities in homography: {torch.isinf(homography).any()}")

		poses2 = poses2.to(device)

		# Concatenate along the last dimension
		embeddings = torch.cat((poses, features, homography, poses2), dim=-1)

		expected_input_size = (self.embed_size * 3) + (self.homog_size * self.num_homog) + self.pose2_size
		print(f"Expected LSTM input size: {expected_input_size}")
		print(f"Actual LSTM input size: {embeddings.shape[-1]}")

		assert embeddings.shape[-1] == expected_input_size, \
			f"Expected input size {expected_input_size}, but got {embeddings.shape[-1]}"

		# Pack the padded sequence
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

		# Check the packed sequence size before feeding into LSTM
		print(f"Packed sequence shape: {packed.data.shape}")

		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs

	def sample(self, features, homography, openpose, states=None):
		sampled_ids = []
		embeddings = torch.zeros([1, 1, self.embed_size]).cuda().float()
		features = features.squeeze(0)

		for i in range(features.shape[0]):
			curr_feat = features[i].unsqueeze(0).unsqueeze(1)
			curr_h = homography[i].unsqueeze(0).unsqueeze(1)
			curr_op = openpose[i].unsqueeze(0).unsqueeze(1)
			tensor = torch.cat((embeddings, curr_feat), 2)
			tensor = torch.cat((tensor, curr_h.cuda()), 2)
			tensor = torch.cat((tensor, curr_op.cuda()), 2)	
			hiddens, states = self.lstm(tensor, states)
			outputs = self.linear(hiddens.squeeze(1))
			_, predicted = outputs.max(1)
			sampled_ids.append(predicted)
			embeddings = self.embed(predicted)
			embeddings = embeddings.unsqueeze(1)
		sampled_ids = torch.stack(sampled_ids, 1)
		return sampled_ids



