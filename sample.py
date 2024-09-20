# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import numpy as np
import argparse 
import pickle
import time
import os
import json
from action_recognition import initialize_actionformer
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

from utils.build_vocab import Vocabulary
from utils.model import EncoderCNN, DecoderRNN
from utils.visualize import show_upp

torch.set_printoptions(threshold=float('inf'))
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_openpose(video_path, openpose_path, seq_length):
	openpose = []
	for i in range(seq_length):
		file_path = os.path.join(openpose_path, "imxx" + str(i + 1) + "_keypoints.json")
		with open(file_path, 'r') as f:
			js = json.load(f)
			if ('people' not in js) or (len(js['people']) <= 0) or ('pose_keypoints_2d' not in js['people'][0]):
				pose2 = [0] * 75
			else:
				pose2 = js['people'][0]['pose_keypoints_2d']
		openpose.append(pose2)
	openpose = torch.Tensor(openpose)
	print("OpenPose Loaded")
	return openpose


def load_homography(video_path, homography_path, seq_length):
	homography = []
	h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 15;
	homography.append(h)
	for i in range(seq_length-1):
		file = open(os.path.join(homography_path, "h" + str(i) + ".txt"))
		h = file.read().split()
		h = map(float, h)
		homography.append(h)
	homography = torch.Tensor(homography)
	print("Homography Loaded")
	return homography


def load_video(video_path, seq_length, transform=None):
	images = []
	for i in range(seq_length):
		image = Image.open(os.path.join(video_path, "imxx" + str(i + 1) + ".jpg")).convert('RGB')
		if transform is not None:
			image = transform(image)
		images.append(image)

	images = torch.stack(images).unsqueeze(0)
	print("Images shape", images.shape)
	return images

def plot_first_10_outputs(outputs):
	# Iterate over the first 10 outputs
	for i in range(10):
		# Extract and reshape each output to [25, 3] for plotting
		joints = outputs[i].view(25, 3).cpu().detach().numpy()

		# Use the provided visualization function
		show_upp(joints)

		# Optionally, pause to control visualization speed
		plt.pause(1)

def main(args):
	actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
	transform = transforms.Compose([
		transforms.Resize(args.crop_size),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225))])

	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	upp_size, low_size = vocab.get_shapes()
	vocab_size = upp_size
	start = time.time()
	encoder = EncoderCNN(args.embed_size, actionformer_feature_extractor).eval()

	decoder = DecoderRNN(args.embed_size,
						 args.hidden_size,
						 upp_size+1,
						 args.seq_length,
						 args.num_layers).to(device)


	decoder.train(False)
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	encoder.load_state_dict(torch.load(args.encoder_path))
	decoder.load_state_dict(torch.load(args.decoder_path))

	video = load_video(args.image_dir, args.seq_length, transform)
	video_tensor = video.to(device)
	feature = encoder(video_tensor)
	homography = load_homography(args.image_dir, args.h_dir, args.seq_length)
	print("Homography shape", homography.shape)
	openpose = load_openpose(args.image_dir, args.openpose_dir, args.seq_length)
	print("Openpose shape", openpose.shape)
	pose_outputs = decoder.sample(feature, homography, openpose)
	print("Pose outputs shape", pose_outputs.shape)
	with open('outputs.txt', 'w') as f:
		f.write(str(pose_outputs))

	# Example usage
	# Assuming 'outputs' is already defined
	plot_first_10_outputs(pose_outputs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab_path', type=str, required=True, help='path for vocabulary wrapper')
	parser.add_argument('--config_path', type=str, default='actionformer/config/anet_tsp.yaml', help='path to the config file')
	parser.add_argument('--output', type=str, required=True, help='output directory to save the pose files to')
	parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
	parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')

	parser.add_argument('--upp', action='store_true', help='set flag if training upper body model')
	parser.add_argument('--low', action='store_true', help='set flag if training lower body model')

	parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect/patty34/synchronized/frames', help='directory for resized images')
	parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect/patty34/features/homography', help='directory for resized images')
	parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect/patty34/features/openpose/output_json', help='directory for resized images')

	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
	parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')
	parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')

	parser.add_argument('--visualize', action='store_true', help='set flag if training lower body model')

	args = parser.parse_args()
	main(args)





