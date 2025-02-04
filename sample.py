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
from utils.model import EncoderCNN, DecoderRNN, TemporalGCN
from utils.visualize import show_upp

torch.set_printoptions(threshold=float('inf'))
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OS:", os.cpu_count())
print("Working On:",device)
# Number of joints and coordinates per joint
num_joints = 25
coords_per_joint = 3

# Define bidirectional connections for the central body part
bidirectional_connections = [
    (0, 1),  # SpineBase <-> SpineMid
    (1, 20), # SpineMid <-> SpineShoulder
    (20, 2), # SpineShoulder <-> Neck
    (2, 3),  # Neck <-> Head
    (0, 12), # SpineBase <-> HipLeft
    (0, 16)  # SpineBase <-> HipRight
]

unidirectional_connections = [
    # Left arm chain
    (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),
    # Right arm chain
    (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),
    # Left leg chain
    (12, 13), (13, 14), (14, 15),
    # Right leg chain
    (16, 17), (17, 18), (18, 19)
]

edge_index = [[], []]

for joint_a, joint_b in bidirectional_connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)
        edge_index[1].append(joint_b * coords_per_joint + j)
        edge_index[0].append(joint_b * coords_per_joint + j)
        edge_index[1].append(joint_a * coords_per_joint + j)

# Add unidirectional edges (one direction for each pair)
for joint_a, joint_b in unidirectional_connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)
        edge_index[1].append(joint_b * coords_per_joint + j)

# Optionally, add self-loops for each joint
for joint in range(num_joints):
    for j in range(coords_per_joint):
        edge_index[0].append(joint * coords_per_joint + j)
        edge_index[1].append(joint * coords_per_joint + j)

edge_index = torch.tensor(edge_index, dtype=torch.long)
print(edge_index.shape)


def load_openpose(video_path, openpose_path, seq_length):
	openpose = []
	for i in range(16,16+seq_length):
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

def load_gt(video_path, gt_path, seq_length):
	gt_pose = []
	for i in range(16,16+seq_length):
		file_path = os.path.join(gt_path, "p" + str(i + 1) + ".txt")
		with open(file_path, 'r') as f:
			egopose_gt = list(map(float, f.read().split()))
		gt_pose.append(egopose_gt)
	gt_pose = torch.Tensor(gt_pose)
	print("Gt Loaded")
	return gt_pose


def load_homography(video_path, homography_path, seq_length):
	homography = []
	h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 15;
	homography.append(h)
	for i in range(16,16+seq_length-1):
		file = open(os.path.join(homography_path, "h" + str(i) + ".txt"))
		h = file.read().split()
		h = map(float, h)
		homography.append(h)
	homography = torch.Tensor(homography)
	print("Homography Loaded")
	return homography


def load_video(video_path, seq_length, transform=None):
	images = []
	for i in range(16,16+seq_length):
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

	start = time.time()
	encoder = EncoderCNN(args.embed_size, actionformer_feature_extractor).to(device)
	temporal_gcn = TemporalGCN(
		args.hidden_size,
		args.seq_length,
		edge_index,
		output_dim=75,
		kernel_size=7).to(device)
	decoder = DecoderRNN(args.embed_size,
						 args.hidden_size,
						 args.seq_length,
						 args.num_layers,
						 temporal_gcn).to(device)


	decoder.train(False)
	encoder.train(False)
	temporal_gcn.train(False)
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
	ground_truth = load_gt(args.image_dir, args.gt_dir, args.seq_length)
	print("Ground truth shape", ground_truth.shape)
	pose_outputs = decoder.sample(feature, homography, openpose)
	print("Pose outputs shape", ground_truth.shape)
	plot_first_10_outputs(pose_outputs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', type=str, default='actionformer/config/anet_tsp.yaml', help='path to the config file')
	parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
	parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')

	parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect/convo47/synchronized/frames', help='directory for resized images')
	parser.add_argument('--gt_dir', type=str, default='you2me_ds_release_kinect/kinect/convo47/synchronized/gt-egopose', help='directory for gt images')
	parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect/convo47/features/homography', help='directory for resized images')
	parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect/convo47/features/openpose/output_json', help='directory for resized images')

	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
	parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')
	parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')

	parser.add_argument('--visualize', action='store_true', help='set flag if training lower body model')

	args = parser.parse_args()
	main(args)





