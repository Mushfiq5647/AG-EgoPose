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
	def __init__(self, embed_size, hidden_size, sequence_length, num_layers, use_homog=True, use_pose2=True, output_size=57,
				 num_homog=15, homog_size=9, pose2_size=50):
		super(DecoderRNN, self).__init__()
		# self.embed = nn.Embedding(vocab_size, embed_size)
		self.use_homog = use_homog
		self.use_pose2 = use_pose2
		self.lstm_input_size = sequence_length + (homog_size * num_homog) + pose2_size
		self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, (output_size))
		# self.embed_size = embed_size
		self.num_homog = num_homog
		self.homog_size = homog_size
		self.pose2_size = pose2_size
		self.output_size = output_size
		self.sequence_length = sequence_length

	def forward(self, features, lengths, homography=None, poses2=None):
		"""Decode image feature vectors and generate pose sequences."""
		device = features.device
		if self.use_homog and self.use_pose2:
			homography = homography.to(device)
			poses2 = poses2.to(device)
			embeddings = torch.cat((features, homography, poses2), dim=-1)
		else:
			embeddings = self.input_projection(features)
		print("Embeddings shape:", embeddings.size(-1))
		print("LSTM size:", self.lstm_input_size)
		# Pack the padded sequence
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
		# Pass through LSTM
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
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



