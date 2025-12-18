import os
import argparse
import pickle
import json
import numpy as np
import cv2
from pathlib import Path

# -------------------------------------------------------------
# Use the same heatmap function as in generate_heatmaps_egopw.py
# -------------------------------------------------------------

def gaussian_heatmaps(joints_xy_img, img_wh, hm_size=64, sigma=2.0):
    W0, H0 = img_wh
    J = joints_xy_img.shape[0]
    Hm = Wm = hm_size
    hm = np.zeros((J, Hm, Wm), dtype=np.float32)

    # precompute Gaussian patch
    r = int(3*sigma)
    d = 2*r + 1
    g = np.arange(d, dtype=np.float32) - r
    xx, yy = np.meshgrid(g, g)
    kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))

    # scale image-pixel joints -> heatmap grid
    sx, sy = Wm / W0, Hm / H0
    jxy = np.stack([joints_xy_img[:,0]*sx, joints_xy_img[:,1]*sy], axis=1)

    for j, (x, y) in enumerate(jxy):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = cx - r, cx + r + 1
        y0, y1 = cy - r, cy + r + 1

        x0c, x1c = max(0, x0), min(Wm, x1)
        y0c, y1c = max(0, y0), min(Hm, y1)
        if x1c <= x0c or y1c <= y0c:
            continue

        kx0, ky0 = x0c - x0, y0c - y0
        patch = kernel[ky0:ky0+(y1c-y0c), kx0:kx0+(x1c-x0c)]
        hm[j, y0c:y1c, x0c:x1c] = np.maximum(hm[j, y0c:y1c, x0c:x1c], patch)

    # normalize each channel (nice for viz)
    mx = hm.reshape(J, -1).max(axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    hm = (hm.reshape(J, -1) / mx).reshape(J, Hm, Wm)
    return hm

# -------------------------------------------------------------
# 3D to 2D projection using fisheye camera (from plot_3d_to_2d.py)
# -------------------------------------------------------------

def load_fisheye_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    size = data.get('size', [1280, 1024])
    intrinsic = np.array(data['intrinsic'], dtype=np.float64)
    polW2C = np.array(data.get('polynomialW2C', []), dtype=np.float64)
    return {
        'W': int(size[0]),
        'H': int(size[1]),
        'intrinsic': intrinsic,
        'polW2C': polW2C
    }

def project_fisheye(pts_3d, fisheye_data):
    """Project 3D points to 2D using fisheye projection"""
    point3d = pts_3d.copy()
    point3d[:, 2] = point3d[:, 2] * -1  # Flip Z as in the reference code
    point3d = point3d.T  # Transpose to 3xN
    
    intrinsic = fisheye_data['intrinsic']
    polW2C = fisheye_data['polW2C'] 
    xc, yc = intrinsic[0, 2], intrinsic[1, 2]  # Image center
    
    point2d_list = []
    
    for i in range(point3d.shape[1]):  # For each point
        x, y, z = point3d[0, i], point3d[1, i], point3d[2, i]
        
        norm = np.sqrt(x*x + y*y)
        
        if norm != 0:
            theta = np.arctan(z / norm)
            invnorm = 1.0 / norm
            t = theta
            rho = polW2C[0]
            t_i = 1.0
            
            for j in range(1, len(polW2C)):
                t_i *= t
                rho += t_i * polW2C[j]
            
            u = x * invnorm * rho + xc
            v = y * invnorm * rho + yc
        else:
            u, v = xc, yc
            
        point2d_list.append([u, v])
    
    return np.array(point2d_list)

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def safe_load_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def to_heatmap_filename(image_name):
    """Convert img_000123.jpg to heatmap_000123.npy"""
    base = os.path.splitext(os.path.basename(image_name))[0]  # e.g., 'img_000100'
    if base.startswith("img_"):
        suffix = base[len("img_"):]
    else:
        suffix = base
    return f"heatmap_{suffix}.npy"

# -------------------------------------------------------------
# Core processing per sequence
# -------------------------------------------------------------

def process_sequence(seq_dir, fisheye_data, expect_joints=15, hm_size=64, sigma=2.0, overwrite=False):
    """Process a single sequence directory"""
    
    # Load annotation data
    pkl_path = os.path.join(seq_dir, "local_pose_gt.pkl")
    if not os.path.isfile(pkl_path):
        print(f"[WARN] No 'local_pose_gt.pkl' in {seq_dir}. Skipping.")
        return 0, 0
    
    # Load synchronization data
    syn_path = os.path.join(seq_dir, "syn.json")
    if not os.path.isfile(syn_path):
        print(f"[WARN] No 'syn.json' in {seq_dir}. Skipping.")
        return 0, 0
    
    # Load data
    annotation_list = safe_load_pickle(pkl_path)
    with open(syn_path, 'r') as f:
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
    
    # Output directory
    out_dir = os.path.join(seq_dir, "heatmap64_2.0")
    os.makedirs(out_dir, exist_ok=True)
    
    # Image directory
    img_dir = os.path.join(seq_dir, "imgs")
    if not os.path.isdir(img_dir):
        print(f"[WARN] No 'imgs' directory in {seq_dir}. Skipping.")
        return 0, 0
    
    saved, total = 0, 0
    
    # Process each annotation entry 
    for annotation in annotation_list:
        total += 1
        
        ext_id = annotation['ext_id']
        ego_id = ext_id - ext_start_frame + ego_start_frame
        ego_image_name = f"img_{ego_id:06d}.jpg"
        
        # Check if image exists
        img_path = os.path.join(img_dir, ego_image_name)
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}. Skipping.")
            continue
        
        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image {img_path}. Skipping.")
            continue
        H0, W0 = img.shape[:2]
        
        # Get 3D pose and project to 2D (use current annotation)
        ego_pose_3d = annotation.get('ego_pose_gt', None)
        if ego_pose_3d is None:
            print(f"[WARN] {ego_image_name}: missing 'ego_pose_gt'. Skipping.")
            continue
        ego_pose_3d = np.asarray(ego_pose_3d)
        if ego_pose_3d.ndim != 2 or ego_pose_3d.shape[1] != 3:
            print(f"[WARN] {ego_image_name}: invalid ego_pose_gt shape {ego_pose_3d.shape}. Skipping.")
            continue
        if ego_pose_3d.shape[0] != expect_joints:
            print(f"[WARN] {ego_image_name}: joints={ego_pose_3d.shape[0]} != expected {expect_joints}. Skipping.")
            continue
        
        # Project 3D pose to 2D using fisheye projection
        joints_2d = project_fisheye(ego_pose_3d, fisheye_data)
        
        # Check if projected points are valid
        valid_mask = np.isfinite(joints_2d).all(axis=1)
        if not valid_mask.any():
            print(f"[WARN] {ego_image_name}: No valid projected joints. Skipping.")
            continue
        
        # Generate heatmaps
        hms = gaussian_heatmaps(joints_2d, (W0, H0), hm_size=hm_size, sigma=sigma)
        if hms.shape != (expect_joints, hm_size, hm_size):
            print(f"[WARN] {ego_image_name}: produced shape {hms.shape} != ({expect_joints},{hm_size},{hm_size}). Skipping save.")
            continue
        
        # Save heatmap
        out_name = to_heatmap_filename(ego_image_name)
        out_path = os.path.join(out_dir, out_name)
        if (not overwrite) and os.path.exists(out_path):
            saved += 1
            continue
        
        np.save(out_path, hms.astype(np.float16))
        saved += 1
        
        if saved % 100 == 0:
            print(f"[INFO] {seq_dir}: Processed {saved}/{total} images...")
    
    return saved, total

# -------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for SceneEgo dataset using 3D->2D projection")
    parser.add_argument("--train_txt", type=str, default='train_sceneego.txt',
                       help="Path to train_unrealego.txt (one sequence directory per line)")
    parser.add_argument("--fisheye_json", type=str, default='2D_projection/fisheye.json',
                       help="Path to fisheye calibration JSON file")
    parser.add_argument("--hm_size", type=int, default=64, help="Heatmap size (square)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma (in heatmap coords)")
    parser.add_argument("--expect_joints", type=int, default=15, help="Expected number of joints")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output .npy exists")
    args = parser.parse_args()
    
    # Load fisheye calibration
    if not os.path.isfile(args.fisheye_json):
        print(f"[ERROR] Fisheye calibration file not found: {args.fisheye_json}")
        return
    
    fisheye_data = load_fisheye_json(args.fisheye_json)
    print(f"[INFO] Loaded fisheye calibration from {args.fisheye_json}")
    print(f"[INFO] Image size: {fisheye_data['W']}x{fisheye_data['H']}")
    
    # Load sequence directories
    if not os.path.isfile(args.train_txt):
        print(f"[ERROR] Train text file not found: {args.train_txt}")
        return
    
    with open(args.train_txt, "r") as f:
        seq_dirs = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    print(f"[INFO] Found {len(seq_dirs)} sequences to process")
    
    total_saved, total_items = 0, 0
    for i, seq in enumerate(seq_dirs):
        if not os.path.isdir(seq):
            print(f"[WARN] Not a directory: {seq}. Skipping.")
            continue
        
        print(f"[SEQ {i+1}/{len(seq_dirs)}] Processing {seq}...")
        
        saved, items = process_sequence(
            seq_dir=seq,
            fisheye_data=fisheye_data,
            expect_joints=args.expect_joints,
            hm_size=args.hm_size,
            sigma=args.sigma,
            overwrite=args.overwrite,
        )
        
        total_saved += saved
        total_items += items
        print(f"[SEQ {i+1}/{len(seq_dirs)}] {seq}: saved {saved}/{items} heatmaps -> {os.path.join(seq, 'heatmap64_2.0')}")
    
    print(f"[ALL] Completed: saved {total_saved}/{total_items} heatmaps total.")

if __name__ == "__main__":
    main()
