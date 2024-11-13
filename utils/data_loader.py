# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import json
import os
import torch
import json
import torch.utils.data as data
from PIL import Image


class PoseDataset(data.Dataset):
	""" Pose custom dataset compatible with torch.utils.data.DataLoader. """
	def __init__(self, annotation, imroot, hroot, oproot, seq_length, test_mode, transform=None):
		self.annotation = annotation
		self.imroot = imroot
		self.hroot = hroot
		self.oproot = oproot
		self.transform = transform
		self.seq_length = seq_length
		self.test_mode = test_mode

	def __getitem__(self, index):
		imroot = self.imroot
		hroot = self.hroot
		oproot = self.oproot
		annotation = self.annotation
		test_mode = self.test_mode
		path, end = annotation.anns[index]
		images = []
		gt_egoposes = []
		poses = []
		poses2 = []
		homography = []
		for i in range(end-self.seq_length, end):
			# print("Image Count", i, end)
			image, gt_egopose = getPair(imroot, hroot, oproot, path, i, test_mode)
			if self.transform is not None:
				image = self.transform(image)
			images.append(image)
			gt_egoposes.append(gt_egopose)
		images = torch.stack(images)
		target_egoposes = torch.Tensor(gt_egoposes)
		return images, target_egoposes

	def __len__(self):
		return len(self.annotation)

def collate_fn(data):
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, target_egoposes = zip(*data)
	images = torch.stack(images, 0)
	lengths = [len(pose) for pose in target_egoposes]
	max_length = max(lengths)
	targets = torch.zeros(len(target_egoposes), max_length, 45)
	for i, pose in enumerate(target_egoposes):
		end = lengths[i]
		targets[i, :end, :] = pose[:end]
	return images, targets, lengths

def getPair(imroot, hroot, oproot, path, index, test_mode):

	gt_data = []
	with open(imroot + "/" + path + "/groundtruth_updated/gt_" + str(index) + ".txt", 'r') as file:
		for line in file:
			# Remove any brackets or extra characters
			line = line.strip().replace('[', '').replace(']', '')

			# Convert the line into a list of floats and append it to data
			row = [float(value) for value in line.split()]
			gt_data.append(row)  # Each row is appended as a list

		# Flatten the 2D list into a 1D list
		egopose_gt = [value for row in gt_data for value in row]

	path = path + "/temp_img_dir/img_" + str(index) + ".jpg"
	image = Image.open(os.path.join(imroot, path)).convert('RGB')
	return image, egopose_gt

def get_loader(annotation, imroot, hroot, oproot, transform, batch_size, shuffle, num_workers, seq_length, test_mode = False):
	""" Returns torch.utils.data.DataLoader for custom pose dataset. """
	ds = PoseDataset(annotation=annotation, imroot=imroot, hroot=hroot, oproot=oproot,
		seq_length=seq_length, transform=transform, test_mode= test_mode)
	data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size,
		shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
	return data_loader