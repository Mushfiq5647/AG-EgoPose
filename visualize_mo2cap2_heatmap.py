import argparse
import os
import h5py
import numpy as np
import cv2


JOINT_NAMES = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder",
    "Left_elbow", "Left_wrist", "Right_hip", "Right_knee", "Right_ankle",
    "Right_foot", "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
]


def _to_hwc_rgb(img_frame: np.ndarray) -> np.ndarray:
    """Convert an image frame of shape [C,H,W], [H,W,C], or [H,W] to H×W×3 RGB float32 in [0,1]."""
    arr = np.asarray(img_frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            # CHW -> HWC
            arr = np.moveaxis(arr, 0, -1)
        # drop alpha if present
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    arr = arr.astype(np.float32)
    # If values look like [0,255], scale to [0,1]
    if arr.max() > 1.5:
        arr = arr / 255.0
    # Ensure within [0,1]
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def overlay_heatmap(rgb_hwc01: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a single heatmap (Hm×Wm in [0,1]) on an RGB image (H×W×3 float in [0,1])."""
    H, W = rgb_hwc01.shape[:2]
    hm = np.asarray(hm, dtype=np.float32)
    # Normalize defensively to [0,1]
    if hm.size > 0:
        mmin, mmax = float(hm.min()), float(hm.max())
        if mmax > mmin:
            hm = (hm - mmin) / (mmax - mmin)
        else:
            hm = np.zeros_like(hm)
    hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap((hm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1.0 - alpha) * rgb_hwc01 + alpha * col
    out = np.clip(out, 0.0, 1.0)
    return out


def overlay_all_heatmaps(rgb_hwc01: np.ndarray, hms: np.ndarray, alpha: float = 0.45, reduce: str = "max") -> np.ndarray:
    """
    Overlay a composite of all joints' heatmaps.
    hms: (J, Hm, Wm) array. Each channel is normalized to [0,1] before reduction.
    reduce: 'max' or 'sum' (sum is clipped to [0,1]).
    """
    J, Hm, Wm = hms.shape
    hms = hms.astype(np.float32)
    # Normalize each joint heatmap independently to [0,1]
    for j in range(J):
        hm = hms[j]
        mmin, mmax = float(hm.min()), float(hm.max())
        if mmax > mmin:
            hms[j] = (hm - mmin) / (mmax - mmin)
        else:
            hms[j] = np.zeros_like(hm, dtype=np.float32)
    if reduce == "sum":
        agg = np.clip(hms.sum(axis=0), 0.0, 1.0)
    else:
        agg = hms.max(axis=0)
    return overlay_heatmap(rgb_hwc01, agg, alpha=alpha)


def grid_visualization(hm_probs: np.ndarray, save_path: str | None):
    """hm_probs: (J, Hm, Wm) in [0,1] — save a 3x5 grid if J==15."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping grid visualization")
        return

    J = hm_probs.shape[0]
    rows = 3 if J == 15 else int(np.ceil(J / 5))
    cols = 5 if J >= 5 else J
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle("Loaded Heatmap Visualizations", fontsize=16)
    for i in range(J):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        im = ax.imshow(hm_probs[i], cmap="hot", interpolation="nearest", vmin=0.0, vmax=1.0)
        name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"Joint_{i}"
        ax.set_title(f"{i}: {name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # hide any unused axes
    for i in range(J, rows*cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved grid to {save_path}")
    plt.show()


def save_overlays(rgb: np.ndarray, hm_probs: np.ndarray, out_dir: str, alpha: float = 0.45):
    os.makedirs(out_dir, exist_ok=True)
    for j in range(hm_probs.shape[0]):
        over = overlay_heatmap(rgb, hm_probs[j], alpha=alpha)
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"joint_{j}"
        out = os.path.join(out_dir, f"overlay_{j:02d}_{name.lower()}.png")
        cv2.imwrite(out, cv2.cvtColor((over*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Saved per-joint overlays to {out_dir}")


def argmax_coords(hm_probs: np.ndarray, orig_hw: tuple[int, int]):
    J, Hm, Wm = hm_probs.shape
    flat = hm_probs.reshape(J, -1)
    idx = flat.argmax(axis=1)
    ys, xs = (idx // Wm).astype(np.float32), (idx % Wm).astype(np.float32)
    H0, W0 = orig_hw
    xs = xs * (W0 / Wm)
    ys = ys * (H0 / Hm)
    coords = np.stack([xs, ys], axis=1)
    conf = hm_probs.max(axis=(1, 2))
    return coords, conf


def save_joint_points(rgb: np.ndarray, hm_probs: np.ndarray, out_dir: str, point_size: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    coords, conf = argmax_coords(hm_probs, rgb.shape[:2])
    img_with_points = (rgb.copy() * 255.0).astype(np.uint8)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 255, 0),
        (255, 192, 203), (0, 128, 0), (128, 128, 0), (255, 20, 147), (70, 130, 180),
    ]
    for j, (coord, confidence) in enumerate(zip(coords, conf)):
        x, y = int(coord[0]), int(coord[1])
        color = colors[j % len(colors)]
        cv2.circle(img_with_points, (x, y), point_size, color, -1)
        cv2.putText(img_with_points, f"{confidence:.2f}", (x + point_size + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out_path = os.path.join(out_dir, "all_joints_points.png")
    cv2.imwrite(out_path, cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR))
    print(f"Saved joint points visualization to {out_path}")
    return coords, conf


def find_heatmap_dataset(f: h5py.File):
    """Return the dataset for heatmaps, supporting 'UpdatedHeatmap' and 'UpdatedHeatMap'."""
    if 'UpdatedHeatmap' in f:
        return f['UpdatedHeatmap'], 'UpdatedHeatmap'
    if 'UpdatedHeatMap' in f:
        return f['UpdatedHeatMap'], 'UpdatedHeatMap'
    # Fall back to common alternatives
    for key in ('Heatmap', 'Heatmaps', 'PredHeatmap', 'PredictedHeatmap'):
        if key in f:
            return f[key], key
    raise KeyError("No heatmap dataset found. Tried 'UpdatedHeatmap' and 'UpdatedHeatMap'.")


def main():
    ap = argparse.ArgumentParser(description="Visualize a single Mo2Cap2 heatmap overlay from HDF5")
    ap.add_argument("--h5", type=str, required=True, help="Path to .h5/.hdf5 file")
    ap.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    ap.add_argument("--joint", type=int, default=-1, help="Joint index [0..14], or -1 for all (default -1)")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    ap.add_argument("--save", type=str, default="./mo2cap2_heatmap_overlay.png", help="Output image path")
    ap.add_argument("--save_dir", type=str, default="./mo2cap2_vis", help="Directory to save additional visualizations")
    ap.add_argument("--save_overlays", action="store_true", help="Save per-joint overlays")
    ap.add_argument("--save_joint_points", action="store_true", help="Save joint points drawn on the image")
    ap.add_argument("--save_grid", action="store_true", help="Save a grid of all heatmaps")
    args = ap.parse_args()

    if not os.path.isfile(args.h5):
        raise FileNotFoundError(args.h5)

    with h5py.File(args.h5, 'r') as f:
        # Locate heatmaps
        dset, dname = find_heatmap_dataset(f)
        if dset.ndim != 4:
            raise ValueError(f"Heatmap dataset '{dname}' has shape {dset.shape}, expected (N,J,H,W)")

        N, J, Hm, Wm = dset.shape
        if not (0 <= args.frame < N):
            raise IndexError(f"frame index {args.frame} out of range [0,{N-1}]")
        if not (args.joint == -1 or (0 <= args.joint < J)):
            raise IndexError(f"joint index {args.joint} out of range [0,{J-1}] (use -1 for all joints)")

        # Fetch image frame
        if 'Images' not in f:
            raise KeyError("Images dataset not found in HDF5. Cannot overlay.")
        img = np.array(f['Images'][args.frame])
        rgb = _to_hwc_rgb(img)

        # Prepare per-frame heatmaps and an RGB image in [0,1]
        hms_all = np.array(dset[args.frame], dtype=np.float32)  # (J,H,W)
        # Normalize per-channel for auxiliary visualizations
        norm_hms = hms_all.copy()
        for j in range(norm_hms.shape[0]):
            hm = norm_hms[j]
            mmin, mmax = float(hm.min()), float(hm.max())
            if mmax > mmin:
                norm_hms[j] = (hm - mmin) / (mmax - mmin)
            else:
                norm_hms[j] = np.zeros_like(hm)

        if args.save_grid:
            grid_path = os.path.join(args.save_dir, "loaded_heatmaps_grid.png")
            grid_visualization(norm_hms, grid_path)

        if args.save_overlays:
            save_overlays(rgb, norm_hms, os.path.join(args.save_dir, "overlays"), alpha=args.alpha)

        if args.save_joint_points:
            coords, conf = save_joint_points(rgb, norm_hms, args.save_dir, point_size=8)
            print("\nJoint coordinates and confidences:")
            for j, (xy, c) in enumerate(zip(coords, conf)):
                name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint_{j}"
                print(f"{j:02d} {name:15s}: (x={xy[0]:.1f}, y={xy[1]:.1f}), conf={c:.3f}")

        if args.joint >= 0:
            # Single-joint overlay
            hm = hms_all[args.joint]
            over = overlay_heatmap(rgb, hm, alpha=args.alpha)
            joint_name = JOINT_NAMES[args.joint] if args.joint < len(JOINT_NAMES) else f"Joint_{args.joint}"
            vis = (over * 255.0).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, f"{args.frame}:{args.joint} {joint_name}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            cv2.imwrite(args.save, vis_bgr)
            print(f"Saved single-joint overlay to {args.save}")
            print(f"Heatmap stats — shape: {(Hm, Wm)}, min/max: {float(hm.min()):.4f}/{float(hm.max()):.4f}")
        else:
            # Composite overlay of all joints
            over = overlay_all_heatmaps(rgb, hms_all, alpha=args.alpha, reduce="max")
            vis = (over * 255.0).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, f"{args.frame}: all joints (max)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            cv2.imwrite(args.save, vis_bgr)
            print(f"Saved composite overlay (all joints) to {args.save}")


if __name__ == "__main__":
    main()


