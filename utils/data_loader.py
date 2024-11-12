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
			image, gt_egopose, h, pose2 = getPair(imroot, hroot, oproot, path, i, test_mode)
			if self.transform is not None:
				image = self.transform(image)
			images.append(image)
			gt_egoposes.append(gt_egopose)
			homography.append(h)
			poses2.append(pose2)
		images = torch.stack(images)
		target_egoposes = torch.Tensor(gt_egoposes)
		homography = [list(h) for h in homography]
		homography = torch.Tensor(homography)
		poses2 = torch.Tensor(poses2)
		return images, target_egoposes, homography, poses2

	def __len__(self):
		return len(self.annotation)

def collate_fn(data):
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, target_egoposes, homography, poses2 = zip(*data)
	images = torch.stack(images, 0)
	lengths = [len(pose) for pose in target_egoposes]
	max_length = max(lengths)
	targets = torch.zeros(len(target_egoposes), max_length, 57)
	for i, pose in enumerate(target_egoposes):
		end = lengths[i]
		targets[i, :end, :] = pose[:end]
	homography = torch.stack(homography, 0)
	poses2 = torch.stack(poses2, 0)
	return images, targets, homography, poses2, lengths

def getPair(imroot, hroot, oproot, path, index, test_mode):
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
				pose2 = [0] * 50
			else:
				pose_keypoints = js['people'][0].get('pose_keypoints_2d', [])

				if len(pose_keypoints) == 0:
					# If no keypoints are found, set to a default value
					pose2 = [0] * 50
				else:
					pose2 = pose_keypoints
					pose2 = [pose2[i] for i in range(len(pose2)) if (i + 1) % 3 != 0]

		with open(imroot + "/" + path + "/synchronized/gt-skeletons/body3DScene_" + str(index) + ".json", 'r') as f:
			js = json.load(f)

			# Check if "bodies" exists and is not empty
			if 'bodies' not in js or len(js['bodies']) == 0:
				# No bodies detected, handle missing data
				egopose_gt = [0] * 57  # 19 joints * 3 coordinates
			else:
				# Find the body with "id" == 0
				body_id_0 = next((body for body in js['bodies'] if body['id'] == 0), None)

				if body_id_0 is None or 'joints19' not in body_id_0:
					# If "id" 0 is not found or no "joints19" data, set to default
					egopose_gt = [0] * 57
				else:
					# Extract 3D coordinates only (skip confidence score)
					joints19 = body_id_0['joints19']
					egopose_gt = [joints19[i] for i in range(len(joints19)) if (i + 1) % 4 != 0]
					egopose_gt_scaled = [round(x / 1000.0, 4) for x in egopose_gt]


	path = path + "/synchronized/frames/imxx" + str(index) + ".jpg"
	image = Image.open(os.path.join(imroot, path)).convert('RGB')
	return image, egopose_gt_scaled, h, pose2

def get_loader(annotation, imroot, hroot, oproot, transform, batch_size, shuffle, num_workers, seq_length, test_mode = False):
	""" Returns torch.utils.data.DataLoader for custom pose dataset. """
	ds = PoseDataset(annotation=annotation, imroot=imroot, hroot=hroot, oproot=oproot,
		seq_length=seq_length, transform=transform, test_mode= test_mode)
	data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size,
		shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
	return data_loader