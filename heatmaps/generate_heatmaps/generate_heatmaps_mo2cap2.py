import os
import argparse
import numpy as np
import h5py

# ---------------------------------------------
# Use THIS heatmap function EXACTLY as provided
# ---------------------------------------------

def gaussian_heatmaps(joints_xy_img, img_wh, hm_size=64, sigma=1.2):
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

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def scan_inputs_from_train(train_txt):
    with open(train_txt, 'r') as f:
        items = [ln.strip() for ln in f if ln.strip()]
    collected = []
    for p in items:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.endswith(('.h5', '.hdf5')):
                        collected.append(os.path.join(root, fn))
        elif os.path.isfile(p) and p.endswith(('.h5', '.hdf5')):
            collected.append(p)
    return sorted(set(collected))


def get_frame_image_size(img_frame):
    shp = img_frame.shape
    if len(shp) == 3 and shp[0] in (1, 3, 4):   # [C,H,W]
        return shp[1], shp[2]
    if len(shp) == 3 and shp[-1] in (1, 3, 4): # [H,W,C]
        return shp[0], shp[1]
    if len(shp) == 2:                           # [H,W]
        return shp[0], shp[1]
    return shp[-2], shp[-1]


# ---------------------------------------------
# Core: write heatmaps back into HDF5 as 'UpdatedHeatmap'
# ---------------------------------------------

def write_heatmaps_inplace(h5_path, hm_size, sigma, offset_x, offset_y, expect_joints=15, overwrite_dataset=True):
    with h5py.File(h5_path, 'a') as f:  # 'a' = read/write, create if needed
        if 'Annot2D' not in f or 'Images' not in f:
            print(f"[WARN] {h5_path}: missing 'Annot2D' or 'Images'. Skipping.")
            return 0

        annot = f['Annot2D']   # [N, J, 2]
        images = f['Images']   # [N, C,H,W] or [N, H,W,C] or [N,H,W]
        N = annot.shape[0]
        J = annot.shape[1]
        if expect_joints and J != expect_joints:
            print(f"[WARN] {h5_path}: joints={J} != expected {expect_joints}. Skipping.")
            return 0

        # Create or refresh dataset
        dset_name = 'UpdatedHeatmap'
        target_shape = (N, J, hm_size, hm_size)
        if dset_name in f:
            existing = f[dset_name]
            if overwrite_dataset or existing.shape != target_shape:
                # delete and recreate
                del f[dset_name]
                dset = f.create_dataset(
                    dset_name, shape=target_shape, dtype=np.float16,
                    chunks=(1, J, hm_size, hm_size), compression='gzip', compression_opts=4
                )
            else:
                dset = existing
        else:
            dset = f.create_dataset(
                dset_name, shape=target_shape, dtype=np.float16,
                chunks=(1, J, hm_size, hm_size), compression='gzip', compression_opts=4
            )

        # Fill per frame
        saved = 0
        for i in range(N):
            joints = np.array(annot[i], dtype=np.float32)  # [J,2]
            if joints.ndim != 2 or joints.shape[0] != J or joints.shape[1] != 2:
                continue

            # size from corresponding image
            img_frame = images[i]
            H0, W0 = get_frame_image_size(img_frame)

            # apply offsets
            joints[:, 0] = joints[:, 0] + float(offset_x)
            joints[:, 1] = joints[:, 1] + float(offset_y)

            # heatmaps
            hms = gaussian_heatmaps(joints, (W0, H0), hm_size=hm_size, sigma=sigma)
            if hms.shape != (J, hm_size, hm_size):
                continue

            dset[i, :, :, :] = hms.astype(np.float16)
            saved += 1

        return saved


def main():
    ap = argparse.ArgumentParser(description="Write (15,128,128) heatmaps back into each HDF5 under dataset 'UpdatedHeatmap'.")
    ap.add_argument('--train_txt', type=str, default='train_mo2cap2.txt', help='Each line: directory or .h5/.hdf5 file')
    ap.add_argument('--hm_size', type=int, default=64, help='Heatmap size (square)')
    ap.add_argument('--sigma', type=float, default=1.2, help='Gaussian sigma in heatmap coords')
    ap.add_argument('--offset_x', type=float, default=-35, help='Offset added to x (pixels)')
    ap.add_argument('--offset_y', type=float, default=5, help='Offset added to y (pixels)')
    ap.add_argument('--expect_joints', type=int, default=15, help='Expected joint count; file is skipped if mismatched')
    ap.add_argument('--overwrite_dataset', action='store_true', help='If set, delete existing UpdatedHeatmap and recreate')
    args = ap.parse_args()

    files = scan_inputs_from_train(args.train_txt)
    if not files:
        print('[ERR] No .h5/.hdf5 files found from train.txt entries.')
        return

    total = 0
    for fp in files:
        saved = write_heatmaps_inplace(
            fp,
            hm_size=args.hm_size,
            sigma=args.sigma,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            expect_joints=args.expect_joints,
            overwrite_dataset=args.overwrite_dataset,
        )
        print(f"[H5] {fp}: wrote {saved} frames into 'UpdatedHeatmap'")
        total += saved

    print(f"[ALL] Done. Total frames written: {total}")


if __name__ == '__main__':
    main()
