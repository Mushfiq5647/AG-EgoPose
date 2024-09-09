# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json
import _compat_pickle
from utils.data_loader import get_loader
from utils.build_vocab import Vocabulary
from utils.build_annotation import Annotation
from utils.model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
torch.set_printoptions(threshold=torch.inf)
# Device configuration
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OS:", os.cpu_count())
print("Working On:",device)
import os



def main(args):

	# create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	# image preprocessing
	transform = transforms.Compose([
		transforms.Resize(args.crop_size),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225))])

	# load vocab wrapper
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)
	print("cluster sizes: ", vocab.get_shapes())
	vocab_dict = {
		'upp_poses': vocab.upp_poses,
		'upp_ids': vocab.upp_ids,
		'low_poses': vocab.low_poses,
		'low_ids': vocab.low_ids
	}


	# Save the JSON string to a .txt file


	print("Contents have been saved to 'vocab_contents.txt'")

	with open(args.annotation_path, 'rb') as f:
		annotation = pickle.load(f)
	print ("annotations size:", len(annotation))

	annotations_dict = {
		"annotations": annotation.anns
	}

	# Save the dictionary to a JSON file
	with open('write.json', 'w') as json_file:
		json.dump(vocab_dict, json_file, indent=4)


	with open('annotation.json', 'w') as json_file:
		json.dump(annotations_dict, json_file, indent=4)

	
	# build data loader
	data_loader = get_loader(annotation, args.image_dir, args.h_dir, args.openpose_dir, vocab, transform, 
		args.batch_size, shuffle=True, num_workers=args.num_workers, seq_length=args.seq_length)
	print("Data Loading complete", len(data_loader))
	print("Total dataset", len(data_loader.dataset))
	upp_size, low_size = vocab.get_shapes()
	vocab_size = upp_size
	encoder = EncoderCNN(args.embed_size).to(device)

	if args.upp:
		model_dir = os.path.abspath('./utils/train_upp')
		decoder = DecoderRNN(args.embed_size, 
							 args.hidden_size, 
							 upp_size+1,
							 upp_size,
							 args.num_layers).to(device)

	elif args.low:
		model_dir = os.path.abspath('./utils/train_low')
		decoder = DecoderRNN(args.embed_size, 
							 args.hidden_size, 
							 vocab_size+1,
							 low_size,
							 args.num_layers).to(device)
	else:
		print('Please specify upper/lower body model to train')
		exit(0)

	# loss and optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)
	# scaler = GradScaler()
	# train the models
	total_step = len(data_loader)
	# print ("total iter", total_step)
	for epoch in range(args.num_epochs):
		print("Printing epoch:", epoch)
		for i, (images, poses, homography, poses2, lengths) in enumerate(data_loader):
			print("Printing iteration number:", i)
			images = images.to(device)
			poses = poses.to(device)
			# if args.upp:
			# 	# Select the first column for upper body pose
			# 	poses = poses[:, :, 0]  # Shape: [batch_size, sequence_length]
			# else:
			# 	# Select the second column for lower body pose
			# 	poses = poses[:, :, 1]  # Shape: [batch_size, sequence_length]


			# Squeeze the targets to make them 1D
			targets = pack_padded_sequence(poses, lengths, batch_first=True)[0]
			targets = targets[:, 1]

			print("In train Pose shape:", poses.shape)
			print("In train Modified targets shape:", targets.shape)
			print("Modified targets:")
			print(targets[0:10])
			with open('sample_targets.txt', 'a') as f:
				f.write(f'{targets[0]}\n')

			print("Before feature")
			# Forward pass
			# with autocast():
			features = encoder(images)
			outputs = decoder(features, poses, homography, poses2, lengths)
			with open('sample_outputs.txt', 'a') as f:
				f.write(f'{outputs[0]}\n')
			# print(f"outputs shape: {outputs.shape}")
			# print(f"targets shape: {targets.shape}")
			# print(outputs[:10])

			# Compute loss
			loss = criterion(outputs, targets)
			if torch.isnan(loss):
				print(f"NaN detected! Epoch: {epoch}, Iteration: {i}")
				# Optionally break the loop or save the current state for debugging
				break

			# Backward and optimize with mixed precision
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_value)
			torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_value)
			optimizer.step()
			if i % args.log_step == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
					  .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

			if ((i + 1) % args.save_step == 0) or (i == total_step - 1):
				torch.save(decoder.state_dict(),
						   os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
				torch.save(encoder.state_dict(),
						   os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))



if __name__== '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
	parser.add_argument('--vocab_path', type=str, required=True, help='path for vocabulary wrapper')
	parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')

	parser.add_argument('--upp', action='store_true', help='set flag if training upper body model')
	parser.add_argument('--low', action='store_true', help='set flag if training lower body model')

	parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for resized images')
	parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for resized images')
	parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for resized images')

	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
	parser.add_argument('--learning_rate', type=float, default=0.0005)
	parser.add_argument('--clip_value', type=float, default=1.0)
	parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')
	parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

	parser.add_argument('--num_epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--num_workers', type=int, default=0)
	
	parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=5, help='step size for saving trained models')
	
	args = parser.parse_args()
	print(args)
	main(args)
