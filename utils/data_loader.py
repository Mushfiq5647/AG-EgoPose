# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import json
from PIL import Image
from utils.build_vocab import Vocabulary
from utils.build_vocab import build_vocab
from utils.build_annotation import Annotation
import os
import argparse


class PoseDataset(data.Dataset):
	""" Pose custom dataset compatible with torch.utils.data.DataLoader. """
	def __init__(self, annotation, imroot, hroot, oproot, vocab, seq_length, test_mode, transform=None):
		self.annotation = annotation
		self.imroot = imroot
		self.hroot = hroot
		self.oproot = oproot
		self.vocab = vocab
		self.transform = transform
		self.seq_length = seq_length
		self.test_mode = test_mode
		# self.data = []
		# self.imroot = os.path.abspath(self.imroot)
		# self.hroot = os.path.abspath(self.hroot)
		# self.oproot = os.path.abspath(self.oproot)
		#
		#
		# for sub_dir in self.train_annotation.train_anns:
		# 	image_dir = os.path.join(self.imroot, sub_dir, 'synchronized','frames')
		# 	gt_dir = os.path.join(self.imroot, sub_dir, 'synchronized', 'gt-egopose')
		# 	print("Image Directory",image_dir, gt_dir)
		# 	self.data.extend(make_dataset(image_dir, gt_dir))


	def __getitem__(self, index):
		imroot = self.imroot
		hroot = self.hroot
		oproot = self.oproot
		vocab = self.vocab
		annotation = self.annotation
		test_mode = self.test_mode
		path, end = annotation.test_anns[index]
		images = []
		gt_egoposes = []
		poses = []
		poses2 = []
		homography = []
		for i in range(end-self.seq_length, end):
			# print("Image Count", i, end)
			image, gt_egopose, h, pose2 = getPair(imroot, hroot, oproot, path, vocab, i, test_mode)
			if self.transform is not None:
				image = self.transform(image)
			images.append(image)
			gt_egoposes.append(gt_egopose)
			homography.append(h)
			poses2.append(pose2)
		images = torch.stack(images)
		target_egoposes = torch.Tensor(gt_egoposes)
		with open('sample_targets.txt', 'a') as f:
			f.write(f'{target_egoposes}\n')
		if homography is not None and all(h is not None for h in homography):
			print("All elements in homography are valid")
			homography = [list(h) for h in homography]
			homography = torch.Tensor(homography)
			poses2 = torch.Tensor(poses2)
		return images, target_egoposes, homography, poses2

	def __len__(self):
		return len(self.annotation)

	# def __getitem__(self, index):
	# 	# Retrieve the sequence of images, ground truth, homography, and OpenPose for the given index
	# 	image_sequence, gt_sequence, homography_sequence, openpose_sequence = self.data[index]
	#
	# 	images = []
	# 	gts = []
	# 	homographies = []
	# 	openposes = []
	#
	# 	# Iterate over each frame in the sequence and load the data
	# 	for image_path, gt_path, homography_path, openpose_path in zip(image_sequence, gt_sequence, homography_sequence,
	# 																   openpose_sequence):
	# 		# Load image
	# 		image = Image.open(image_path)
	#
	# 		# Load ground truth
	# 		with open(gt_path, 'r') as f:
	# 			gt = f.read()  # Process the ground truth as necessary
	#
	# 		# Load homography
	# 		with open(homography_path, 'r') as f:
	# 			homography = f.read().split()  # Process homography if needed
	# 			homography = list(map(float, homography))  # Convert to floats
	#
	# 		# Load OpenPose data (assuming JSON format)
	# 		with open(openpose_path, 'r') as f:
	# 			openpose = json.load(f)
	# 			pose = openpose.get('joints', [0] * 48)  # Default to 48 zeros if 'joints' is not present
	#
	# 		# Append the loaded data to the lists
	# 		images.append(image)
	# 		gts.append(gt)
	# 		homographies.append(homography)
	# 		openposes.append(pose)
	#
	# 	# Convert lists to tensors
	# 	images = torch.stack([self.transform(image) if self.transform else image for image in
	# 						  images])  # Apply transform and stack images
	# 	gts = torch.Tensor(gts)  # Convert ground truth to tensor
	# 	homographies = torch.Tensor(homographies)  # Convert homographies to tensor
	# 	openposes = torch.Tensor(openposes)  # Convert OpenPose data to tensor
	#
	# 	# Return the sequence of images, ground truth, homographies, and OpenPose data
	# 	return images, gts, homographies, openposes


def collate_fn(data):
	""" Creates mini-batch tensors from the list of tuples (images, poses) """
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, target_egoposes, homography, poses2 = zip(*data)
	print("Pose length:", len(target_egoposes))
	# with open("sample_poes
	# 		f.write(str(pose) + '\n')
		# print("Image length:", len(images))
	images = torch.stack(images, 0)
	lengths = [len(pose) for pose in target_egoposes]
	max_length = max(lengths)
	# Now a 3D tensor to hold [batch_size, sequence_length, 2]
	targets = torch.zeros(len(target_egoposes), max_length, 75)

	print("Pose2 shape:", len(target_egoposes))
	print("Max Length:", max_length)
	for i, pose in enumerate(target_egoposes):
		end = lengths[i]
		targets[i, :end, :] = pose[:end]
	if isinstance(homography, torch.Tensor) and isinstance(poses2, torch.Tensor):
		homography = torch.stack(homography, 0)
		# with open('sample_targets.txt', 'w') as f:
		# 	for i in range(min(10, targets.shape[0])):  # limiting to the first 10 samples for brevity
		# 		f.write(f"Sample {i + 1}:\n")
		# 		f.write(f"{targets[i]}\n")
		# 		f.write("\n")
		poses2 = torch.stack(poses2, 0)

	return images, targets, homography, poses2, lengths

def getPair(imroot, hroot, oproot, path, vocab, index, test_mode):
	""" helper method to get the image corresponding to the pair """
	if not test_mode:
		if index <= 1:
			h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 15
		else:
			file = open(hroot + "/" + path + "/features/homography/h" + str(index - 2) + ".txt")
			h = file.read().split()
			h = map(float, h)

		with open(oproot + "/" + path + "/features/openpose/output_json/imxx" + str(index) + "_keypoints.json", 'r') as f:
			js = json.loads(f.read())
			if 'people' not in js or len(js['people']) == 0:
				# No people detected, handle missing data
				pose2 = [0] * 75  # Default to 75 zeros (assuming 25 joints * 3 values: x, y, confidence)
			else:
				# Extract the keypoints for the first person in the 'people' array
				pose_keypoints = js['people'][0].get('pose_keypoints_2d', [])

				if len(pose_keypoints) == 0:
					# If no keypoints are found, set to a default value
					pose2 = [0] * 75  # Default to 75 zeros (25 joints * 3 values: x, y, confidence)
				else:
					# If keypoints are found, use them
					pose2 = pose_keypoints
	else:
		h = None
		pose2 = None

	egopose_file = imroot + "/" + path + "/synchronized/gt-egopose/p" + str(index) + ".txt"
	with open(egopose_file, 'r') as f:
		egopose_gt = list(map(float, f.read().split()))
		# print("Egopose:",egopose_file, egopose_gt)
		# Convert to a list of 75 float values
	# print("Upper Cluster:", upp_cluster)
	# print("Lower Cluster:", low_cluster)
	path = path + "/synchronized/frames/imxx" + str(index) + ".jpg"
	image = Image.open(os.path.join(imroot, path)).convert('RGB')
	return image, egopose_gt, h, pose2

# def make_dataset(image_dir, gt_dir, h_dir, oproot, seq_length):
#     data = []
#     image_dir = os.path.normpath(image_dir)
#     gt_dir = os.path.normpath(gt_dir)
#     h_dir = os.path.normpath(h_dir)
#     oproot = os.path.normpath(oproot)
#
#     if os.path.exists(image_dir) and os.path.exists(gt_dir):
#         images = sorted(os.listdir(image_dir))
#         gts = sorted(os.listdir(gt_dir))
#         homographies = sorted(os.listdir(h_dir))
#         openposes = sorted(os.listdir(oproot))
#
#         # Iterate over the images, ground truth, homography, and OpenPose data in steps of seq_length
#         for i in range(0, len(images) - seq_length + 1, seq_length):
#             image_sequence = images[i:i + seq_length]
#             gt_sequence = gts[i:i + seq_length]
#             homography_sequence = homographies[i:i + seq_length]
#             openpose_sequence = openposes[i:i + seq_length]
#
#             # Store the entire sequence as one entry in the list
#             data.append((image_sequence, gt_sequence, homography_sequence, openpose_sequence))
#
#     return data


def get_loader(annotation, imroot, hroot, oproot, vocab, transform, batch_size, shuffle, num_workers, seq_length, test_mode = False):
	""" Returns torch.utils.data.DataLoader for custom pose dataset. """
	ds = PoseDataset(annotation=annotation, imroot=imroot, hroot=hroot, oproot=oproot, vocab=vocab,
		seq_length=seq_length, transform=transform, test_mode= test_mode)
	data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size,
		shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
	return data_loader