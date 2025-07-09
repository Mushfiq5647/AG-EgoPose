import os, glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import json
from PIL import Image
from natsort import natsorted
from utils.image_folder import make_dataset
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        image /= self.std

        return image


def dataloader_full(opt, transform, mode='train', id=None):
    if mode == 'train':
        shuffle = True
        drop_last = True
    elif mode == 'validation':
        shuffle = False
        drop_last = False
    elif mode == 'test':
        shuffle = False
        drop_last = False

    if opt.model == "egoglass":
        datasets = UnrealEgoStereoWindowDataset(opt, transform, mode, id=id)
    elif opt.model == "unrealego_heatmap_shared":
        datasets = CreateStereoHeatmapDataset(opt, mode, id=id)
    elif opt.model == "unrealego_autoencoder":
        datasets = UnrealEgoStereoWindowDataset(opt, mode, id=id)

    dataset = torch.utils.data.DataLoader(
        datasets,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
        collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    print("Data Loading complete", len(dataset))
    return dataset


class UnrealEgoStereoWindowDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, id=None, window_size=32, stride=16, pad=True):
        self.opt = opt
        self.mode = mode
        self.data_list_path = os.path.join(opt.data_dir, mode + '.txt')  # CT\UnrealEgo\static00\UnrealEgoData\train.txt
        with open(self.data_list_path, 'r') as f:
            seq_dirs = [line.strip() for line in f if line.strip()]
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.id = id
        self.pad = pad

        # Build index of (seq_idx, start)
        self.sequences = []  # list of sorted npy file lists
        self.homography_sequences = []  # list of sorted npy file lists
        self.index = []      # list of (seq_idx, start)

        for seq_idx, base_dir in enumerate(seq_dirs):
            npy_dir = os.path.join(base_dir, "all_data_with_img-256_hm-64_pose-16_npy")
            homography_dir = os.path.join(base_dir, "normalized_homography")
            files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
            homography_files = sorted([f for f in os.listdir(homography_dir) if f.endswith('left.txt')])
            full_paths = [os.path.join(npy_dir, f) for f in files]
            full_homography_paths = [os.path.join(homography_dir, f) for f in homography_files]
            self.sequences.append(full_paths)
            self.homography_sequences.append(full_homography_paths)


            L = len(full_paths)
            L_f = len(full_paths)
            assert L == L_f, "Lengths must be equal"
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, start))
            if pad:
                if L < window_size:
                    self.index.append((seq_idx, 0))
                elif (L - window_size) % stride != 0:
                    self.index.append((seq_idx, L - window_size))
        # print("Total Index", self.index)
        print("Total data", len(self.index))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start = self.index[idx]
        files = self.sequences[seq_idx]
        homography_files = self.homography_sequences[seq_idx]
        L = len(files)
        lf = len(homography_files)
        # print("Image length and homography length", L, lf)
        if L!=lf:
            raise AssertionError(f"Lengths must be equal", L, lf, homography_files, idx)

        end = start + self.window_size

        # Select window with padding if needed
        if end <= L:
            window = files[start:end]
            h_window = homography_files[start:end]
        else:
            tail = files[start:]
            h_tail = homography_files[start:]
            pad_count = end - L
            tail += [files[-1]] * pad_count
            h_tail += [homography_files[-1]] * pad_count
            window = tail
            h_window = h_tail

        left_imgs, right_imgs, poses, homographies = [], [], [], []
        for path in window:
            data = np.load(path, allow_pickle=True).item()
            left  = torch.from_numpy(data['input_rgb_left']).float()
            right = torch.from_numpy(data['input_rgb_right']).float()
            left = self.transform(left)
            right = self.transform(right)
            pose = torch.from_numpy(data['gt_local_pose']).float()
            left_imgs.append(left)
            right_imgs.append(right)
            poses.append(pose)

        for homography_path in h_window:
            H = np.loadtxt(homography_path)  # shape (3,3)
            homographies.append(H)

        if len(h_window)!=len(window):
            print(f"Idx {idx}: h_window has length {len(h_window)}")
            raise AssertionError(f"Windows must be equal")

        if not homographies:
            raise RuntimeError(f"No homography data for sample index {idx} (seq {seq_idx}, start {start})")
        # Instead of torch.Tensor(homographies), first stack into a single NumPy array
        oh_array = np.stack(homographies, axis=0)  # shape (T,3,3)

        # Then convert to a tensor in one go
        homography_batch = torch.from_numpy(oh_array).float()

        # Stack into tensors
        left_batch  = torch.stack(left_imgs,  dim=0)  # (T,3,H,W)
        right_batch = torch.stack(right_imgs, dim=0)  # (T,3,H,W)
        pose_batch  = torch.stack(poses, dim=0)  # (T,P)
        # print("Image", left_batch.shape)

        return {
            'input_rgb_left':  left_batch,
            # 'input_rgb_right': right_batch,
            'input_homography': homography_batch,
            'gt_local_pose':   pose_batch
        }

from typing import Dict, Any

def custom_collate_fn(batch: list[Dict[str, Any]]):
    max_len = 0
    min_len = float('inf')
    for sample in batch:
        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                seq_len = v.size(0)
                max_len = max(max_len, seq_len)
                min_len = min(min_len, seq_len)
    # print(f"Batch sequence lengths => min: {min_len}, max: {max_len}")

    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        vals = [sample[key] for sample in batch]

        # Sequence-like tensor fields
        if isinstance(vals[0], torch.Tensor) and vals[0].dim() >= 1:
            padded = []
            for seq in vals:
                L = seq.size(0)
                # Debug mismatch for specific keys
                if L != max_len and key in ("input_rgb_left", "input_homography", 'gt_local_pose'):
                    print(f"Key '{key}' has mismatched length: {L}")
                # Pad shorter sequences by repeating last timestep
                if L < max_len:
                    pad_amt = max_len - L
                    last = seq[-1].unsqueeze(0).expand(pad_amt, *seq.shape[1:])
                    seq = torch.cat([seq, last], dim=0)
                padded.append(seq)
            out[key] = torch.stack(padded, dim=0)

        # Static tensor fields
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)

        # Non-tensor fields (e.g., file paths)
        else:
            out[key] = vals

    return out

# Usage:
# from torch.utils.data import DataLoader
# loader = DataLoader(
#     dataset,
#     batch_size=opt.batch_size,
#     shuffle=True,
#     num_workers=opt.num_threads,
#     collate_fn=custom_collate_fn,
#     drop_last=drop_last
# )


class CreateStereoHeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode, id=None):
        super(CreateStereoHeatmapDataset, self).__init__()
        self.opt = opt
        self.load_size_rgb = opt.load_size_rgb
        self.load_size_heatmap = opt.load_size_heatmap
        self.data_list_path = os.path.join(opt.data_dir, mode + '.txt')
        self.frame_data_paths, self.num_frame_data = make_dataset(
            opt=opt,
            data_list_path=self.data_list_path,
            data_sub_path='all_data_with_img-256_hm-64_pose-16_npy',
            id=id
        )

    def __getitem__(self, index):
        # get paths for each data
        frame_data_path = self.frame_data_paths[index]

        # load each data
        frame_data = np.load(frame_data_path, allow_pickle=True)
        frame_data = frame_data.item()

        input_rgb_left = torch.from_numpy(frame_data["input_rgb_left"]).float()
        input_rgb_right = torch.from_numpy(frame_data["input_rgb_right"]).float()
        gt_heatmap_left = torch.from_numpy(frame_data["gt_heatmap_left"]).float()
        gt_heatmap_right = torch.from_numpy(frame_data["gt_heatmap_right"]).float()

        return {"frame_data_path": frame_data_path,
                'input_rgb_left': input_rgb_left,
                'input_rgb_right': input_rgb_right,
                'gt_heatmap_left': gt_heatmap_left,
                'gt_heatmap_right': gt_heatmap_right,
                }

    def __len__(self):
        return self.num_frame_data

if __name__ == "__main__":
    img = T.ToPILImage()(torch.randn(3, 224, 224))
    color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    transform = T.ColorJitter.get_params(
        color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
        color_jitter.hue)

    img_trans1 = transform(img)
    img_trans2 = transform(img)
    print((np.array(img_trans1) == np.array(img_trans2)).all())
