import os
import argparse
import pickle
import numpy as np
import cv2

# -------------------------------------------------------------
# Use THIS heatmap function EXACTLY as provided by the user
# -------------------------------------------------------------

def gaussian_heatmaps(joints_xy_img, img_wh, hm_size=128, sigma=2.0):
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
# Helpers
# -------------------------------------------------------------

def safe_load_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def get_joints2d(entry):
    # accept both 'joints_2d' and 'joints_2D'
    if isinstance(entry, dict):
        if "joints_2d" in entry:
            arr = entry["joints_2d"]
        elif "joints_2D" in entry:
            arr = entry["joints_2D"]
        else:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr
    return None


def resolve_image_path(seq_dir, image_name, img_subdir="imgs"):
    p1 = os.path.join(seq_dir, img_subdir, image_name)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(seq_dir, image_name)
    if os.path.isfile(p2):
        return p2
    return None


def to_heatmap_filename(image_name):
    base = os.path.splitext(os.path.basename(image_name))[0]  # e.g., 'img_000100'
    if base.startswith("img_"):
        suffix = base[len("img_"):]
    else:
        suffix = base
    return f"heatmap_{suffix}.npy"

# -------------------------------------------------------------
# Core processing per sequence
# -------------------------------------------------------------

def process_sequence(seq_dir, expect_joints, hm_size, sigma, overwrite=False, img_subdir="imgs"):
    pkl_path = os.path.join(seq_dir, "pseudo.pkl")  # per user: exact name
    if not os.path.isfile(pkl_path):
        # small convenience fallback in case some dirs used the older name
        alt = os.path.join(seq_dir, "pseudo_gt.pkl")
        if os.path.isfile(alt):
            pkl_path = alt
        else:
            print(f"[WARN] No 'pseudo.pkl' in {seq_dir}. Skipping.")
            return 0, 0

    data = safe_load_pickle(pkl_path)
    if not isinstance(data, (list, tuple)):
        print(f"[WARN] Unexpected PKL format in {pkl_path}. Expected a list of dicts. Skipping.")
        return 0, 0

    # Output directory is STRICTLY '<seq_dir>/heatmap'
    out_dir = os.path.join(seq_dir, "heatmap")
    os.makedirs(out_dir, exist_ok=True)

    saved, total = 0, 0

    for idx, entry in enumerate(data):
        total += 1
        if not isinstance(entry, dict):
            print(f"[WARN] Entry {idx} in {pkl_path} is not a dict. Skipping.")
            continue

        img_name = entry.get("image_name", None)
        if not img_name:
            print(f"[WARN] Missing 'image_name' for entry {idx} in {pkl_path}. Skipping.")
            continue

        joints = get_joints2d(entry)
        if joints is None:
            print(f"[WARN] Missing/invalid joints_2d for {img_name}. Skipping.")
            continue
        if joints.shape[0] != expect_joints:
            print(f"[WARN] {img_name}: joints={joints.shape[0]} != expected {expect_joints}. Skipping.")
            continue

        img_path = resolve_image_path(seq_dir, img_name, img_subdir=img_subdir)
        if img_path is None:
            print(f"[WARN] Image not found for {img_name} under {seq_dir}. Skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image {img_path}. Skipping.")
            continue
        H0, W0 = img.shape[:2]

        # Generate heatmaps with the EXACT function above
        hms = gaussian_heatmaps(joints, (W0, H0), hm_size=hm_size, sigma=sigma)
        if hms.shape != (expect_joints, hm_size, hm_size):
            print(f"[WARN] {img_name}: produced shape {hms.shape} != ({expect_joints},{hm_size},{hm_size}). Skipping save.")
            continue

        out_name = to_heatmap_filename(img_name)  # e.g., 'heatmap_000100.npy'
        out_path = os.path.join(out_dir, out_name)
        if (not overwrite) and os.path.exists(out_path):
            saved += 1
            continue

        np.save(out_path, hms.astype(np.float16))
        saved += 1

    return saved, total


# -------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate (15,128,128) heatmaps and save to '<seq>/heatmap/heatmap_XXXXXX.npy'.")
    parser.add_argument("--train_txt", type=str, default='train_egopw.txt', help="Path to train.txt (one sequence directory per line)")
    parser.add_argument("--hm_size", type=int, default=128, help="Heatmap size (square). Must be 128 for your spec.")
    parser.add_argument("--sigma", type=float, default=2, help="Gaussian sigma (in heatmap coords).")
    parser.add_argument("--expect_joints", type=int, default=15, help="Expected number of joints (default: 15)")
    parser.add_argument("--img_subdir", type=str, default="imgs", help="Subfolder inside each sequence with images (default: 'imgs')")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output .npy exists")
    args = parser.parse_args()

    if args.hm_size != 128:
        print(f"[INFO] You set hm_size={args.hm_size}. Requirement says 128x128; continuing anyway.")

    # Load sequence directories
    with open(args.train_txt, "r") as f:
        seq_dirs = [ln.strip() for ln in f.readlines() if ln.strip()]

    total_saved, total_items = 0, 0
    for seq in seq_dirs:
        if not os.path.isdir(seq):
            print(f"[WARN] Not a directory: {seq}. Skipping.")
            continue
        saved, items = process_sequence(
            seq_dir=seq,
            expect_joints=args.expect_joints,
            hm_size=args.hm_size,
            sigma=args.sigma,
            overwrite=args.overwrite,
            img_subdir=args.img_subdir,
        )
        total_saved += saved
        total_items += items
        print(f"[SEQ] {seq}: saved {saved}/{items} heatmaps -> {os.path.join(seq, 'heatmap')}")

    print(f"[ALL] Completed: saved {total_saved}/{total_items} heatmaps total.")


if __name__ == "__main__":
    main()
