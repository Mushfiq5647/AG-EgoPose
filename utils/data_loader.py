import os, glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pickle as pkl
import h5py
import scipy.io
import json
from PIL import Image
from natsort import natsorted
from utils.image_folder import make_dataset
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
TEST_DATASET = "EgoGlobal"


def resolve_data_file_path(filename):
    if os.path.exists(filename):
        return filename
    else:
        # Look in parent directory
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
        if os.path.exists(parent_path):
            return parent_path
        else:
            raise FileNotFoundError(f"Could not find {filename} in current directory or parent directory")


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

    if opt.model == "mo2cap2":
        datasets = Mo2Cap2WindowDataset(opt, transform, mode)
    elif opt.model == "unrealego_heatmap_shared":
        datasets = CreateStereoHeatmapDataset(opt, transform, mode, id=id)
    elif opt.model == "egopw":
        datasets = EgoPwWindowDataset(opt, transform, mode)
    elif opt.model == "sceneego":
        datasets = SceneEgoWindowDataset(opt, transform, mode)
    elif opt.model == "unrealego":
        datasets = UnrealEgoWindowDataset(opt, transform, mode)
    elif opt.model == "egoglobal":
        datasets = EgoGlobalTestDataset(opt, transform, mode)
    elif opt.model == "egogta":
        datasets = EgoGTAWindowDataset(opt, transform, mode)
    elif opt.model == "combined":
        ds1 = EgoPwWindowDataset(opt, transform, mode)
        ds2 = SceneEgoWindowDataset(opt, transform, mode)
        datasets = ConcatDataset([ds1, ds2])
    dataset = torch.utils.data.DataLoader(
        datasets,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
        # collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    print("Data Loading complete", len(dataset))
    return dataset


class Mo2Cap2WindowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,          # one .hdf5 path per line
                 transform,         # torchvision transform for RGB frames
                 mode,
                 window_size=64,
                 stride = 32):
        super().__init__()
        self.transform   = transform
        self.window_size = window_size
        self.stride      = stride

        h5_list_txt = f'{mode}_mo2cap2.txt'

        # 1) read all .hdf5 paths
        file_path = resolve_data_file_path(h5_list_txt)
        with open(file_path, 'r') as f:
            raw_paths = [l.strip() for l in f if l.strip()]

        # 2) filter out any that immediately fail to open
        good_paths = []
        for p in raw_paths:
            try:
                with h5py.File(p, 'r'):
                    pass
                good_paths.append(p)
            except OSError as e:
                print(f"⚠️  Skipping corrupted file {p}: {e}")
        self.h5_paths = good_paths

        # 3) build our sliding‑window index
        self.index = []
        for seq_idx, p in enumerate(self.h5_paths):
            with h5py.File(p, 'r') as f:
                N = f['Images'].shape[0]
            for start in range(0, N - self.window_size + 1, self.stride):
                self.index.append((seq_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start = self.index[idx]
        h5_path = self.h5_paths[seq_idx]

        # 4) try to read our window; if it fails, skip this sample
        try:
            with h5py.File(h5_path, 'r') as f:
                imgs_np  = f['Images'   ][start:start+self.window_size]  # uint8
                hms_np   = f['UpdatedHeatmap' ][start:start+self.window_size]  # uint8
                poses_np = f['Annot3D'  ][start:start+self.window_size]  # float64
        except OSError as e:
            print(f"⚠️  I/O error at {h5_path}[{start}:{start+self.window_size}]: {e}")
            raise IndexError  # DataLoader will skip this sample

        # 5) to torch, normalize images+heatmaps to [0,1], keep poses as-is
        imgs  = torch.from_numpy(imgs_np.astype(np.float32) / 255.0)  # (T,3,256,256)
        hms   = torch.from_numpy(hms_np) # (T,J,32,32)
        poses = torch.from_numpy(poses_np.astype(np.float32))  / 1000.0  # mm → meters   # (T,J,3)

        # 6) optionally apply your standard RGB transforms per frame
        if self.transform is not None:
            out = []
            for t in range(imgs.shape[0]):
                # back to PIL.Image for the same pipeline you use with EgoPW
                frame = (imgs[t].permute(1,2,0).numpy() * 255).astype(np.uint8)
                pil   = Image.fromarray(frame)
                out.append(self.transform(pil))
            imgs = torch.stack(out, dim=0)  # now (T,3,H',W')

        return {
            'input_rgb':    imgs,   # (T,3,H,W)
            # 'gt_heatmap':   hms,    # (T,J,32,32)
            'gt_local_pose':poses   # (T,J,3)
        }


class EgoPwWindowDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, window_size=64, stride=64):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        # Read sequence directories
        sequnce_directory = f'{mode}_egopw.txt'
        print(sequnce_directory)
        file_path = resolve_data_file_path(sequnce_directory)
        with open(file_path, 'r') as f:
            seq_dirs = [l.strip() for l in f if l.strip()]

        self.sequences = []   # list of (file_paths, heatmap_dir)
        self.gt_dicts  = []   # list of ground-truth maps
        self.index     = []   # list of (seq_idx, start_idx)

        for seq_idx, base in enumerate(seq_dirs):
            img_dir = os.path.join(base, 'imgs')
            hm_dir = os.path.join(base, 'heatmap64_2.0')

            # Load ground-truth list and build map
            with open(os.path.join(base, 'pseudo_gt.pkl'), 'rb') as f:
                gt_list = pkl.load(f)
            gt_map = {d['image_name']: d for d in gt_list}

            # 1) Identify and remove corrupted images
            bad_imgs = {
                name for name, info in gt_map.items()
                if np.isnan(info['optimized_local_pose']).any()
            }
            if bad_imgs:
                print(f"⚠️ Sequence {base}: removing {len(bad_imgs)} corrupt images")

            # 2) List and filter jpg files
            files = natsorted(
                [f for f in os.listdir(img_dir)
                 if f.endswith('.jpg') and f not in bad_imgs]
            )

            if bad_imgs:
                before = len(gt_map)
                gt_map = {k: v for k, v in gt_map.items() if k not in bad_imgs}
                # print(f"   → pruned GT: {before} → {len(gt_map)}")

            # 3) Build full paths and store
            full_paths = [os.path.join(img_dir, f) for f in files]
            self.sequences.append((full_paths))
            self.gt_dicts.append(gt_map)
            # print("Image count", len(full_paths))
            # print("Gt count", len(gt_map))

            # 4) Create sliding-window indices on filtered list
            L = len(full_paths)
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, hm_dir, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, hm_dir, start = self.index[idx]
        img_paths = self.sequences[seq_idx]
        gt_map = self.gt_dicts[seq_idx]

        # Extract one window of valid images only
        window = img_paths[start:start + self.window_size]

        imgs, hms, poses = [], [], []
        for p in window:
            name = os.path.basename(p)
            if name.startswith("img_"):
                suffix = name[len("img_"):]
            else:
                suffix = name

            info = gt_map[name]
            # Load and convert pose (guaranteed no NaN)
            pose_np = info['optimized_local_pose']
            poses.append(torch.from_numpy(pose_np).float())

            # Load image
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

            # Load precomputed heatmap
            heatmap_filename = "heatmap_" + suffix.replace('.jpg', '.npy')
            hm_path = os.path.join(hm_dir, heatmap_filename)
            hm = np.load(hm_path)
            hms.append(torch.from_numpy(hm).float())

        # Stack into tensors
        img_batch  = torch.stack(imgs,  dim=0)  # (T,3,H,W)
        hm_batch   = torch.stack(hms,  dim=0)   # (T,J,H,W)
        pose_batch = torch.stack(poses, dim=0)  # (T,J,3)

        return {
            'input_rgb':    img_batch,
            'gt_heatmap':   hm_batch,
            'gt_local_pose':pose_batch
        }


class SceneEgoWindowDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, window_size=128, stride=32):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        # Read sequence directories
        sequnce_directory = f'{mode}_sceneego.txt'
        print(sequnce_directory)
        file_path = resolve_data_file_path(sequnce_directory)
        with open(file_path, 'r') as f:
            seq_dirs = [l.strip() for l in f if l.strip()]

        self.sequences = []   # list of (file_paths, heatmap_dir)
        self.gt_dicts  = []   # list of ground-truth maps
        self.index     = []   # list of (seq_idx, start_idx)

        for seq_idx, base in enumerate(seq_dirs):
            img_dir = os.path.join(base, 'imgs')
            hm_dir = os.path.join(base, 'heatmap64_2.0')

            # Load annotation data
            with open(os.path.join(base, 'local_pose_gt.pkl'), 'rb') as f:
                annotation_list = pkl.load(f)
            
            # Load synchronization data
            with open(os.path.join(base, 'syn.json'), 'r') as f:
                syn_data = json.load(f)
            
            ego_start_frame = syn_data['ego']
            ext_start_frame = syn_data['ext']
            
            # Create mapping from external ID to egocentric image name and pose
            gt_map = {}
            for annotation in annotation_list:
                ext_id = annotation['ext_id']
                ego_id = ext_id - ext_start_frame + ego_start_frame
                ego_image_name = f"img_{ego_id:06d}.jpg"
                
                # Store the ego_pose_gt as the pose data
                gt_map[ego_image_name] = {
                    'ego_pose_gt': annotation['ego_pose_gt'],
                    'ext_id': ext_id,
                    'ego_id': ego_id
                }
            
            print(f"Sequence {base}: {len(annotation_list)} annotations mapped to {len(gt_map)} egocentric images")
            
            # 1) Identify and remove corrupted images
            bad_imgs = set()
            for name, info in gt_map.items():
                try:
                    pose_array = np.array(info['ego_pose_gt'], dtype=float)
                    if np.isnan(pose_array).any():
                        bad_imgs.add(name)
                except (ValueError, TypeError):
                    # Handle non-numeric data
                    bad_imgs.add(name)
            if bad_imgs:
                print(f"⚠️ Sequence {base}: removing {len(bad_imgs)} corrupt images")
                gt_map = {k: v for k, v in gt_map.items() if k not in bad_imgs}

            # 2) List all jpg files in the directory
            all_files = natsorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            print(f"Sequence {base}: {len(all_files)} total images in directory")
            
            # 3) Filter to only images that have ground truth
            valid_files = [f for f in all_files if f in gt_map]
            print(f"Sequence {base}: {len(valid_files)} images with ground truth")

            # 4) Build full paths and store
            full_paths = [os.path.join(img_dir, f) for f in valid_files]
            self.sequences.append((full_paths))
            self.gt_dicts.append(gt_map)

            # 5) Create sliding-window indices on filtered list
            L = len(full_paths)
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, hm_dir, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, hm_dir, start = self.index[idx]
        img_paths = self.sequences[seq_idx]
        gt_map = self.gt_dicts[seq_idx]

        # Extract one window of valid images only
        window = img_paths[start:start + self.window_size]

        imgs, hms, poses = [], [], []
        for p in window:
            name = os.path.basename(p)
            if name.startswith("img_"):
                suffix = name[len("img_"):]
            else:
                suffix = name
            
            info = gt_map[name]
            # Load and convert pose (guaranteed no NaN)
            pose_np = np.array(info['ego_pose_gt'])  # Use ego_pose_gt and convert to numpy array
            poses.append(torch.from_numpy(pose_np).float())

            # Load image
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

            heatmap_filename = "heatmap_" + suffix.replace('.jpg', '.npy')
            hm_path = os.path.join(hm_dir, heatmap_filename)
            hm = np.load(hm_path)
            hms.append(torch.from_numpy(hm).float())

        # Stack into tensors
        img_batch  = torch.stack(imgs,  dim=0)  # (T,3,H,W)
        hm_batch = torch.stack(hms, dim=0)  # (T,J,H,W)
        pose_batch = torch.stack(poses, dim=0)  # (T,J,3)

        return {
            'input_rgb':    img_batch,
            'gt_heatmap': hm_batch,
            'gt_local_pose':pose_batch
        }


class EgoGTAWindowDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, window_size=64, stride=32):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        # Read sequence directories
        sequnce_directory = f'{mode}_egogta.txt'
        print(sequnce_directory)
        file_path = resolve_data_file_path(sequnce_directory)
        with open(file_path, 'r') as f:
            seq_dirs = [l.strip() for l in f if l.strip()]

        self.sequences = []  # list of (image_paths, heatmap_paths, pose_array)
        self.index = []  # list of (seq_idx, start_idx) for sliding windows

        for seq_idx, base in enumerate(seq_dirs):
            # list and sort image files
            img_dir = os.path.join(base, 'img')
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
            img_files = natsorted(img_files)
            img_paths = [os.path.join(img_dir, f) for f in img_files]

            pkl_path = os.path.join(base, 'joints_3d.pkl')
            with open(pkl_path, 'rb') as f:
                pose_arr = pkl.load(f)  # shape (N, J, 3)
            # print(len(pose_arr))
            assert isinstance(pose_arr, np.ndarray), f"Expected numpy.ndarray, got {type(pose_arr)}"

            # Handle mismatch between poses and images by truncating to the shorter length
            if pose_arr.shape[0] != len(img_paths):
                min_length = min(pose_arr.shape[0], len(img_paths))
                print(
                    f"⚠️  Mismatch in {base}: {pose_arr.shape[0]} poses vs {len(img_paths)} images. Truncating to {min_length}")
                pose_arr = pose_arr[:min_length]
                img_paths = img_paths[:min_length]

            self.sequences.append((img_paths, pose_arr))
            L = len(img_paths)
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start = self.index[idx]
        img_paths, pose_arr = self.sequences[seq_idx]

        window_imgs = img_paths[start:start + self.window_size]
        # window_hms  = hm_paths[start:start + self.window_size]
        window_poses = pose_arr[start:start + self.window_size]  # (T, J, 3)

        imgs, hms, poses = [], [], []
        for img_path, pose in zip(window_imgs, window_poses):
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

            # load heatmap
            # hm = np.load(os.path.join(hm_dir, f"heatmaps_{suffix.replace('.jpg','.npy')}"))
            # hms.append(torch.from_numpy(hm).float())

            # load pose
            poses.append(torch.from_numpy(pose).float())

        img_batch = torch.stack(imgs, dim=0)  # (T,3,H,W)
        # hm_batch   = torch.stack(hms,  dim=0)      # (T,J,H,W)
        pose_batch = torch.stack(poses, dim=0)  # (T,J,3)

        return {
            'input_rgb': img_batch,
            # 'gt_heatmap':   hm_batch,
            'gt_local_pose': pose_batch
        }


class UnrealEgoWindowDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, window_size=64, stride=16):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        # Read sequence directories
        sequnce_directory = f'{mode}_unrealego.txt'
        print(sequnce_directory)
        file_path = resolve_data_file_path(sequnce_directory)
        with open(file_path, 'r') as f:
            seq_dirs = [l.strip() for l in f if l.strip()]

        self.sequences = []   # list of (file_paths, heatmap_dir)
        self.index = []

        for seq_idx, base in enumerate(seq_dirs):
            # list and sort image files
            sequence_dir = os.path.join(base, 'all_data_with_img-256_hm-64_pose-16_npy')
            npy_files = [f for f in os.listdir(sequence_dir) if f.lower().endswith(('.npy'))]
            npy_files = natsorted(npy_files)
            npy_paths = [os.path.join(sequence_dir, f) for f in npy_files]

            self.sequences.append(npy_paths)
            L = len(npy_paths)
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # get paths for each data
        seq_idx, start = self.index[idx]
        npy_paths = self.sequences[seq_idx]

        window_files = npy_paths[start:start + self.window_size]
        # window_hms  = hm_paths[start:start + self.window_size]

        imgs_left, imgs_right, hms_left, hms_right, poses = [], [], [], [], []
        for frame_data_path in window_files:
            # load each data
            frame_data = np.load(frame_data_path, allow_pickle=True)
            frame_data = frame_data.item()

            input_rgb_left = frame_data["input_rgb_left"]
            input_rgb_right = frame_data["input_rgb_right"]

            if self.transform:
                # Debug: Check dimensions
                # print(f"DEBUG - input_rgb_left shape: {input_rgb_left.shape}, dtype: {input_rgb_left.dtype}")
                # print(f"DEBUG - input_rgb_right shape: {input_rgb_right.shape}, dtype: {input_rgb_right.dtype}")
                
                # Convert from (C, H, W) to (H, W, C) and float32 to uint8
                input_rgb_left = np.transpose(input_rgb_left, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                input_rgb_right = np.transpose(input_rgb_right, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                
                # print(f"DEBUG - After transpose - left shape: {input_rgb_left.shape}, right shape: {input_rgb_right.shape}")
                
                # Convert to PIL Image
                input_rgb_left = Image.fromarray((input_rgb_left * 255).astype(np.uint8))
                input_rgb_right = Image.fromarray((input_rgb_right * 255).astype(np.uint8))

                input_rgb_left = self.transform(input_rgb_left)
                input_rgb_right = self.transform(input_rgb_right)
            gt_heatmap_left = torch.from_numpy(frame_data["gt_heatmap_left"]).float()
            gt_heatmap_right = torch.from_numpy(frame_data["gt_heatmap_right"]).float()
            gt_local_pose = torch.from_numpy(frame_data["gt_local_pose"]).float()
            imgs_left.append(input_rgb_left)
            imgs_right.append(input_rgb_right)
            hms_left.append(gt_heatmap_left)
            hms_right.append(gt_heatmap_right)
            poses.append(gt_local_pose)


        img_left_batch = torch.stack(imgs_left, dim=0)  # (T,3,H,W)
        img_right_batch = torch.stack(imgs_right, dim=0)  # (T,3,H,W)
        hm_left_batch   = torch.stack(hms_left,  dim=0)      # (T,J,H,W)
        hm_right_batch   = torch.stack(hms_right,  dim=0)      # (T,J,H,W)
        pose_batch = torch.stack(poses, dim=0)  # (T,J,3)

        return {
            'input_rgb_left': img_left_batch,
            'input_rgb_right': img_right_batch,
            'gt_heatmap_left':   hm_left_batch,
            'gt_heatmap_right':   hm_right_batch,
            'gt_local_pose': pose_batch
        }

class EgoGlobalTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform, mode, window_size=64, stride=32):
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        # Read sequence directories
        if TEST_DATASET == "EgoGlobal":
            sequnce_directory = f'{mode}_egoglobal.txt'
        else:
            sequnce_directory = f'{mode}_mo2cap2.txt'
        
        file_path = resolve_data_file_path(sequnce_directory)
        with open(file_path, 'r') as f:
            seq_dirs = [l.strip() for l in f if l.strip()]

        self.sequences = []  # list of (image_paths, heatmap_paths, pose_array)
        self.index = []      # list of (seq_idx, start_idx) for sliding windows

        for seq_idx, base in enumerate(seq_dirs):
            # list and sort image files
            img_dir = os.path.join(base, 'imgs')
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
            img_files = natsorted(img_files)
            img_paths = [os.path.join(img_dir, f) for f in img_files]

            # list and sort corresponding heatmap files
            # hm_dir = os.path.join(base, 'heatmaps')
            # hm_files = [f.replace('.jpg', '_heatmaps.npy') for f in img_files]
            # hm_paths = [os.path.join(hm_dir, f) for f in hm_files]

            # load pose numpy array
            seq_name = os.path.basename(base.rstrip('/'))
            if TEST_DATASET=="EgoGlobal":
                pkl_path = os.path.join(base, f"{seq_name}.pkl")
                with open(pkl_path, 'rb') as f:
                    pose_arr = pkl.load(f)  # shape (N, J, 3)

            else:
                mat_path = os.path.join(base, f"{seq_name}_gt.mat")
                pose_data = scipy.io.loadmat(mat_path)
                pose_arr = pose_data['pose_gt'] / 1000.0  # Convert from mm to cm


            assert isinstance(pose_arr, np.ndarray), f"Expected numpy.ndarray, got {type(pose_arr)}"
            
            # Handle mismatch between poses and images by truncating to the shorter length
            if pose_arr.shape[0] != len(img_paths):
                min_length = min(pose_arr.shape[0], len(img_paths))
                print(f"⚠️  Mismatch in {base}: {pose_arr.shape[0]} poses vs {len(img_paths)} images. Truncating to {min_length}")
                pose_arr = pose_arr[:min_length]
                img_paths = img_paths[:min_length]

            self.sequences.append((img_paths, pose_arr))
            L = len(img_paths)
            for start in range(0, L - window_size + 1, stride):
                self.index.append((seq_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start = self.index[idx]
        img_paths, pose_arr = self.sequences[seq_idx]

        window_imgs = img_paths[start:start + self.window_size]
        # window_hms  = hm_paths[start:start + self.window_size]
        window_poses = pose_arr[start:start + self.window_size]  # (T, J, 3)

        imgs, hms, poses = [], [], []
        for img_path, pose in zip(window_imgs, window_poses):
            name = os.path.basename(img_path)
            if name.startswith("img_"):
                suffix = name[len("img_"):]
            else:
                suffix = name
            # load and transform image
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

            # load heatmap
            # hm = np.load(os.path.join(hm_dir, f"heatmaps_{suffix.replace('.jpg','.npy')}"))
            # hms.append(torch.from_numpy(hm).float())

            # load pose
            poses.append(torch.from_numpy(pose).float())

        img_batch  = torch.stack(imgs, dim=0)      # (T,3,H,W)
        # hm_batch   = torch.stack(hms,  dim=0)      # (T,J,H,W)
        pose_batch = torch.stack(poses, dim=0)     # (T,J,3)

        return {
            'input_rgb':    img_batch,
            # 'gt_heatmap':   hm_batch,
            'gt_local_pose':pose_batch
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
