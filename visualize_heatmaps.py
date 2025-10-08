import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt

# If your project exposes options, you can import and parse them here.
# Otherwise, comment the next two lines.
try:
    from options.train_options import TrainOptions
    _HAS_OPTS = True
except Exception:
    _HAS_OPTS = False
    TrainOptions = None

# Import your heatmap network
try:
    from heatmaps.network_heatmap import HeatMap_Network  # project-relative import
except Exception:
    # Fallback import if the class lives at repo root as `network_heatmap.py`
    from heatmaps.network_heatmap import HeatMap_Network

JOINT_NAMES = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder",
    "Left_elbow", "Left_wrist", "Right_hip", "Right_knee", "Right_ankle",
    "Right_foot", "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
]

# -----------------------------
# Model / I/O helpers
# -----------------------------

def build_model(num_joints: int, ckpt_path: str, device: str = "cuda", model_name: str = "resnet18"):
    # If your HeatMap_Network expects TrainOptions, pass them; else create a dummy one.
    if _HAS_OPTS and TrainOptions is not None:
        # Create a dummy opt without parsing command line args
        class DummyOpt:
            def __init__(self):
                self.num_heatmap = 15
                self.init_ImageNet = True
        opt = DummyOpt()
    else:
        class DummyOpt:
            def __init__(self):
                self.num_heatmap = 15
                self.init_ImageNet = True
        opt = DummyOpt()
    model = HeatMap_Network(opt, model_name='resnet18')

    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both plain state_dict and wrapped dicts
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    return model


def get_preprocess(img_size: int):
    # Match training preprocessing (update if you used different stats)
    return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str, img_size: int, device: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    
    # Keep original image for overlay
    original_rgb = rgb.copy()
    
    # Resize for model input
    x = get_preprocess(img_size)(rgb).unsqueeze(0).to(device)
    return x, original_rgb, (H0, W0)


def overlay_heatmap(rgb: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a single heatmap (Hm×Wm in [0,1]) on an RGB image (H×W×3)."""
    H, W = rgb.shape[:2]
    hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap((hm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(rgb, 1.0, col, alpha, 0)
    return out


def predict_heatmaps(model, x: torch.Tensor):
    with torch.no_grad():
        logits = model(x)                # (B, J, Hm, Wm) logits
        probs = torch.sigmoid(logits)    # (B, J, Hm, Wm) in [0,1] for visualization
    return logits, probs


# -----------------------------
# Visualization
# -----------------------------

def grid_visualization(hm_probs: np.ndarray, save_path: str | None):
    """hm_probs: (J, Hm, Wm) in [0,1]"""
    J = hm_probs.shape[0]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Predicted Heatmap Visualizations", fontsize=16)
    for i in range(J):
        r, c = divmod(i, 5)
        im = axes[r, c].imshow(hm_probs[i], cmap="hot", interpolation="nearest", vmin=0.0, vmax=1.0)
        axes[r, c].set_title(f"{i}: {JOINT_NAMES[i]}")
        axes[r, c].axis("off")
        plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved grid to {save_path}")
    plt.show()


def save_overlays(rgb: np.ndarray, hm_probs: np.ndarray, out_dir: str, alpha: float = 0.45):
    os.makedirs(out_dir, exist_ok=True)
    for j in range(hm_probs.shape[0]):
        over = overlay_heatmap(rgb, hm_probs[j], alpha=alpha)
        out = os.path.join(out_dir, f"overlay_{j:02d}_{JOINT_NAMES[j].lower()}.png")
        cv2.imwrite(out, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    print(f"Saved per-joint overlays to {out_dir}")


def save_joint_points(rgb: np.ndarray, hm_probs: np.ndarray, out_dir: str, point_size: int = 8):
    """Save images with joint points marked on the original image"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Get joint coordinates from heatmaps
    coords, conf = argmax_coords(hm_probs, rgb.shape[:2])
    
    # Create image with joint points
    img_with_points = rgb.copy()
    
    # Define colors for different joints (BGR format for OpenCV)
    colors = [
        (255, 0, 0),    # Red - Neck
        (0, 255, 0),    # Green - Right shoulder
        (0, 0, 255),    # Blue - Right elbow
        (255, 255, 0),  # Cyan - Right wrist
        (255, 0, 255),  # Magenta - Left shoulder
        (0, 255, 255),  # Yellow - Left elbow
        (128, 0, 128),  # Purple - Left wrist
        (255, 165, 0),  # Orange - Right hip
        (0, 128, 255),  # Light blue - Right knee
        (128, 255, 0),  # Lime - Right ankle
        (255, 192, 203), # Pink - Right foot
        (0, 128, 0),    # Dark green - Left hip
        (128, 128, 0),  # Olive - Left knee
        (255, 20, 147), # Deep pink - Left ankle
        (70, 130, 180), # Steel blue - Left foot
    ]
    
    # Draw joint points
    for j, (coord, confidence) in enumerate(zip(coords, conf)):
        x, y = int(coord[0]), int(coord[1])
        color = colors[j]
        
        # Draw circle for joint point
        cv2.circle(img_with_points, (x, y), point_size, color, -1)
        
        # Draw confidence text
        cv2.putText(img_with_points, f"{confidence:.2f}", 
                   (x + point_size + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
    
    # Save the image with all joint points
    out_path = os.path.join(out_dir, "all_joints_points.png")
    cv2.imwrite(out_path, cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR))
    print(f"Saved joint points visualization to {out_path}")
    
    # Also save individual joint point images
    for j, (coord, confidence) in enumerate(zip(coords, conf)):
        x, y = int(coord[0]), int(coord[1])
        color = colors[j]
        
        # Create individual image with just this joint
        individual_img = rgb.copy()
        cv2.circle(individual_img, (x, y), point_size, color, -1)
        cv2.putText(individual_img, f"{JOINT_NAMES[j]} ({confidence:.2f})", 
                   (x + point_size + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        out_path = os.path.join(out_dir, f"joint_{j:02d}_{JOINT_NAMES[j].lower()}_point.png")
        cv2.imwrite(out_path, cv2.cvtColor(individual_img, cv2.COLOR_RGB2BGR))
    
    print(f"Saved individual joint point images to {out_dir}")
    return coords, conf


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


# -----------------------------
# CLI
# -----------------------------

def main():
    # Create a separate parser to avoid conflicts with TrainOptions
    import sys
    original_argv = sys.argv.copy()
    
    # Filter out any conflicting arguments
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--image', '--checkpoint', '--img_size', '--num_joints', '--save_dir', '--save_overlays', '--save_joint_points', '--alpha', '--point_size']:
            filtered_argv.append(arg)
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                filtered_argv.append(sys.argv[i + 1])
                i += 1
        i += 1
    
    sys.argv = filtered_argv
    
    ap = argparse.ArgumentParser(description="Predict & visualize heatmaps with a pretrained model")
    ap.add_argument("--image", type=str, default='/data/My_Backup/Dataset/SceneEgo_test/diogo1/imgs/img_000059.jpg', help="Path to input image")
    ap.add_argument("--checkpoint", type=str, default='utils/trained_heatmaps/bce_combined/heatmap_best.ckpt', help="Path to model .pth/.pt state_dict")
    ap.add_argument("--img_size", type=int, default=256, help="Input size used in training (e.g., 224)")
    ap.add_argument("--num_joints", type=int, default=15, help="Number of joints")
    ap.add_argument("--save_dir", type=str, default='./overlay_out', help="Where to save visualizations")
    ap.add_argument("--save_overlays", action="store_true", help="Also save per-joint heatmap overlays on the RGB")
    ap.add_argument("--save_joint_points", action="store_true", help="Save images with joint points marked")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for heatmaps")
    ap.add_argument("--point_size", type=int, default=8, help="Size of joint points")
    args = ap.parse_args()

    # Restore original argv
    sys.argv = original_argv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.num_joints, args.checkpoint, device, model_name='resnet18')

    x, rgb, orig_hw = load_image(args.image, args.img_size, device)
    logits, probs = predict_heatmaps(model, x)

    # Log quick stats
    # print(f"Pred logits range: {logits.min().item():.3f}..{logits.max().item():.3f}")
    print(f"Pred probs range : {probs.min().item():.3f}..{probs.max().item():.3f}")

    # Grid of probabilities
    grid_out = os.path.join(args.save_dir, "pred_heatmaps_grid.png") if args.save_dir else None
    grid_visualization(probs[0].cpu().numpy(), grid_out)

    # Optional overlays per joint
    if args.save_overlays and args.save_dir is not None:
        save_overlays(rgb, probs[0].cpu().numpy(), args.save_dir, alpha=args.alpha)

    # Get joint coordinates from heatmaps
    coords, conf = argmax_coords(probs[0].cpu().numpy(), orig_hw)
    
    # Always save joint points on original image
    if args.save_dir is not None:
        save_joint_points(rgb, probs[0].cpu().numpy(), args.save_dir, point_size=args.point_size)
    
    for j, (xy, c) in enumerate(zip(coords, conf)):
        print(f"{j:02d} {JOINT_NAMES[j]:15s}: (x={xy[0]:.1f}, y={xy[1]:.1f}), conf={c:.3f}")

if __name__ == "__main__":
    main()
