#!/usr/bin/env python3
import os
import numpy as np
import cv2
from scipy.linalg import svd
from tqdm import tqdm

# === Settings ===
TEST_LIST = '/data/My_Backup/UnrealEgo/scripts/data/UnrealEgoData/test.txt'  # path to your train.txt
IMG_SUBFOLDER = 'all_data_with_img-256_hm-64_pose-16_npy'
HOMO_SUBFOLDER = 'homography'
# Normalization range for denormalization
min_val, max_val = -2.1179, 2.6400

# === Helpers ===
def denormalize_img(img: np.ndarray) -> np.ndarray:
    """Convert normalized float image to uint8 [0,255]."""
    img = np.clip((img - min_val) / (max_val - min_val) * 255, 0, 255)
    return img.astype(np.uint8)


def get_point_correspondences(img1: np.ndarray, img2: np.ndarray):
    """Detect ORB keypoints, match, and return corresponding pts or (None,None)."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Compute 3x3 homography H using DLT + SVD."""
    N = pts1.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float32)
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2*i]   = [ x1,  y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2 ]
        A[2*i+1] = [  0,   0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2 ]
    _, _, vh = svd(A)
    h = vh[-1].reshape(3,3)
    return h / h[0,0]


# Identity fallback matrix
eye3 = np.eye(3, dtype=np.float32)

# === Main ===
with open(TEST_LIST, 'r') as f:
    seq_paths = [line.strip() for line in f if line.strip()]

for seq in tqdm(seq_paths, desc='Sequences'):
    img_dir  = os.path.join(seq, IMG_SUBFOLDER)
    homo_dir = os.path.join(seq, HOMO_SUBFOLDER)
    if not os.path.isdir(img_dir):
        print(f"[WARN] Missing image folder: {img_dir}")
        continue
    os.makedirs(homo_dir, exist_ok=True)

    # Collect and sort frame files
    frames = sorted(
        [fn for fn in os.listdir(img_dir) if fn.endswith('.npy')],
        key=lambda fn: int(os.path.splitext(fn)[0].split('_')[-1])
    )
    N = len(frames)
    if N == 0:
        print(f"[WARN] No .npy frames in {img_dir}")
        continue

    # Write h0 for both sides as identity
    for side in ('left', 'right'):
        np.savetxt(os.path.join(homo_dir, f'h0_{side}.txt'), eye3, fmt='%.6f')

    # Compute homographies for indices 1..N-1
    for i in range(N-1):
        idx = i + 1
        # Load frame i and i+1
        data0 = np.load(os.path.join(img_dir, frames[i]), allow_pickle=True).item()
        data1 = np.load(os.path.join(img_dir, frames[i+1]), allow_pickle=True).item()

        # LEFT
        img0 = denormalize_img(data0['input_rgb_left'].transpose(1,2,0))
        img1 = denormalize_img(data1['input_rgb_left'].transpose(1,2,0))
        pts1, pts2 = get_point_correspondences(img0, img1)
        H_left = estimate_homography(pts1, pts2) if pts1 is not None else eye3
        np.savetxt(os.path.join(homo_dir, f'h{idx}_left.txt'), H_left, fmt='%.6f')

        # RIGHT
        img0r = denormalize_img(data0['input_rgb_right'].transpose(1,2,0))
        img1r = denormalize_img(data1['input_rgb_right'].transpose(1,2,0))
        pts1r, pts2r = get_point_correspondences(img0r, img1r)
        H_right = estimate_homography(pts1r, pts2r) if pts1r is not None else eye3
        np.savetxt(os.path.join(homo_dir, f'h{idx}_right.txt'), H_right, fmt='%.6f')


OUTPUT_SUBFOLDER = 'normalized_homography'
PERCENTILE = 80

# Identity matrix
eye3 = np.eye(3, dtype=np.float32)


# === Helpers ===
def load_homography(file_path):
    return np.loadtxt(file_path, dtype=np.float32)


def save_homography(file_path, H):
    np.savetxt(file_path, H, fmt='%.6f')


# === Process All Sequences ===
with open(TEST_LIST, 'r') as f:
    seq_paths = [line.strip() for line in f if line.strip()]

for seq in tqdm(seq_paths, desc='Normalizing Sequences'):
    homo_dir = os.path.join(seq, HOMO_SUBFOLDER)
    out_dir = os.path.join(seq, OUTPUT_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)

    # Collect all homography file names
    homo_files = [fn for fn in os.listdir(homo_dir) if fn.endswith('.txt')]

    # Collect all deltas
    all_deltas = []
    for fn in homo_files:
        H = load_homography(os.path.join(homo_dir, fn))
        delta = np.abs(H - eye3)
        all_deltas.append(delta.flatten())

    all_deltas = np.concatenate(all_deltas)
    scale = np.percentile(all_deltas, PERCENTILE)

    # Avoid dividing by zero
    if scale == 0:
        scale = 1.0

    for fn in homo_files:
        H = load_homography(os.path.join(homo_dir, fn))
        delta = H - eye3
        H_scaled = delta / scale
        save_homography(os.path.join(out_dir, fn), H_scaled)
