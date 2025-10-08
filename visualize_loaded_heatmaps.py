import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

JOINT_NAMES = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder",
    "Left_elbow", "Left_wrist", "Right_hip", "Right_knee", "Right_ankle",
    "Right_foot", "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
]

def load_heatmap(heatmap_path: str):
    """Load a pre-generated heatmap file (.npy)"""
    if not os.path.exists(heatmap_path):
        raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")
    
    heatmaps = np.load(heatmap_path)
    print(f"Loaded heatmaps shape: {heatmaps.shape}")
    print(f"Heatmaps range: {heatmaps.min():.4f} to {heatmaps.max():.4f}")
    print(f"Heatmaps mean: {heatmaps.mean():.4f}")
    
    return heatmaps

def load_image(image_path: str):
    """Load the corresponding image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    print(f"Loaded image shape: {rgb.shape}")
    
    return rgb, (H0, W0)

def overlay_heatmap(rgb: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a single heatmap (Hm×Wm in [0,1]) on an RGB image (H×W×3)."""
    H, W = rgb.shape[:2]
    hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap((hm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(rgb, 1.0, col, alpha, 0)
    return out

def grid_visualization(hm_probs: np.ndarray, save_path: str = None):
    """hm_probs: (J, Hm, Wm) in [0,1]"""
    J = hm_probs.shape[0]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Loaded Heatmap Visualizations", fontsize=16)
    
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
    """Save individual heatmap overlays"""
    os.makedirs(out_dir, exist_ok=True)
    
    for j in range(hm_probs.shape[0]):
        over = overlay_heatmap(rgb, hm_probs[j], alpha=alpha)
        out = os.path.join(out_dir, f"overlay_{j:02d}_{JOINT_NAMES[j].lower()}.png")
        cv2.imwrite(out, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    
    print(f"Saved per-joint overlays to {out_dir}")

def argmax_coords(hm_probs: np.ndarray, orig_hw: tuple[int, int]):
    """Extract joint coordinates from heatmaps"""
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

def create_merged_heatmap(
    hm_probs: np.ndarray,
    save_path: str = None,
    use_different_colors: bool = True,
    blend_mode: str = "max",           # "max" | "sum"
    sharpen_gamma: float = 2.0,         # >1 sharpens peaks
    threshold: float = 0.15,            # suppress low values (fraction of per-joint max)
    alpha: float = 1.0                  # per-joint alpha when blending colors
):
    """
    Merge all 15 joint heatmaps into a single image with distinct colors for each joint.
    
    Args:
        hm_probs: (J, Hm, Wm) heatmaps for all joints
        save_path: Path to save the merged visualization
        use_different_colors: Whether to use different colors for each joint
    """
    J, Hm, Wm = hm_probs.shape
    
    if use_different_colors:
        # Create a merged heatmap with different colors for each joint
        merged = np.zeros((Hm, Wm, 3), dtype=np.float32)

        # All white colors for 15 joints (RGB in [0,1])
        colors = [
            (1.0, 1.0, 1.0),  # Neck - white
            (1.0, 1.0, 1.0),  # R shoulder - white
            (1.0, 1.0, 1.0),  # R elbow - white
            (1.0, 1.0, 1.0),  # R wrist - white
            (1.0, 1.0, 1.0),  # L shoulder - white
            (1.0, 1.0, 1.0),  # L elbow - white
            (1.0, 1.0, 1.0),  # L wrist - white
            (1.0, 1.0, 1.0),  # R hip - white
            (1.0, 1.0, 1.0),  # R knee - white
            (1.0, 1.0, 1.0),  # R ankle - white
            (1.0, 1.0, 1.0),  # R foot - white
            (1.0, 1.0, 1.0),  # L hip - white
            (1.0, 1.0, 1.0),  # L knee - white
            (1.0, 1.0, 1.0),  # L ankle - white
            (1.0, 1.0, 1.0),  # L foot - white
        ]

        # Blend heatmaps with thresholding and sharpening to reduce blur
        merged_sum = np.zeros_like(merged)
        merged_max = np.zeros_like(merged)

        for j in range(J):
            hm_j = hm_probs[j]

            # Per-joint normalization (guard against zero)
            max_j = float(hm_j.max()) if hm_j.max() > 0 else 1.0
            hm_j = hm_j / max_j

            # Threshold low responses to suppress haze
            if threshold > 0:
                t = threshold
                hm_j = np.clip((hm_j - t) / max(1e-6, 1.0 - t), 0.0, 1.0)

            # Sharpen peaks
            if sharpen_gamma is not None and sharpen_gamma > 1.0:
                hm_j = np.power(hm_j, sharpen_gamma)

            color = np.array(colors[j], dtype=np.float32).reshape(1, 1, 3)
            colored = (hm_j[:, :, None] * color) * alpha

            merged_sum += colored
            merged_max = np.maximum(merged_max, colored)

        merged_sum = np.clip(merged_sum, 0.0, 1.0)
        merged_max = np.clip(merged_max, 0.0, 1.0)

        merged = merged_max if blend_mode == "max" else merged_sum
        
    else:
        # Simple max operation - take maximum across all joints
        merged = np.max(hm_probs, axis=0)  # (Hm, Wm)
        # Convert to RGB by replicating the channel
        merged = np.stack([merged, merged, merged], axis=-1)  # (Hm, Wm, 3)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if use_different_colors:
        ax.imshow(merged, interpolation='nearest')
        ax.set_title('Merged Heatmap - All 15 Joints (Crisp Colors)', fontsize=16, fontweight='bold')
        
        # Add legend showing joint colors
        legend_elements = []
        for j in range(J):
            color = colors[j]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=f"{j}: {JOINT_NAMES[j]}"))
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    else:
        ax.imshow(merged[:, :, 0], cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title('Merged Heatmap - All 15 Joints (Max Intensity)', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Max Joint Probability', fontsize=12)
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved merged heatmap to {save_path}")
    
    plt.show()
    return merged

def create_body_pose_features_heatmap(hm_probs: np.ndarray, rgb_image: np.ndarray = None, save_path: str = None):
    """
    Create a 'Body Pose Features' style heatmap similar to the reference image.
    This creates a colorful, semi-transparent overlay showing all joint activations.
    Optionally overlays on the original image.
    """
    J, Hm, Wm = hm_probs.shape
    
    # Create a colorful merged heatmap with distinct colors for each joint
    # Using a vibrant color palette similar to the reference image
    colors = [
        (0.2, 0.8, 0.2),   # Bright green - Neck
        (0.8, 0.8, 0.2),   # Yellow - Right shoulder
        (0.8, 0.4, 0.2),   # Orange - Right elbow
        (0.8, 0.2, 0.8),   # Magenta - Right wrist
        (0.2, 0.8, 0.8),   # Cyan - Left shoulder
        (0.4, 0.2, 0.8),   # Purple - Left elbow
        (0.8, 0.2, 0.2),   # Red - Left wrist
        (0.2, 0.4, 0.8),   # Blue - Right hip
        (0.6, 0.8, 0.2),   # Lime - Right knee
        (0.8, 0.6, 0.2),   # Gold - Right ankle
        (0.2, 0.6, 0.8),   # Light blue - Right foot
        (0.6, 0.2, 0.8),   # Violet - Left hip
        (0.8, 0.8, 0.6),   # Light yellow - Left knee
        (0.6, 0.4, 0.8),   # Lavender - Left ankle
        (0.4, 0.8, 0.6),   # Mint - Left foot
    ]
    
    # Create the merged heatmap with smooth blending
    merged_rgb = np.zeros((Hm, Wm, 3), dtype=np.float32)
    
    for j in range(J):
        hm_j = hm_probs[j]
        
        # Normalize the heatmap for this joint
        max_val = float(hm_j.max()) if hm_j.max() > 0 else 1.0
        hm_normalized = hm_j / max_val
        
        # Apply smooth Gaussian-like blending for softer appearance
        # Use a softer gamma correction
        hm_smooth = np.power(hm_normalized, 0.8)
        
        # Apply soft thresholding to create smoother gradients
        threshold = 0.05
        hm_smooth = np.clip((hm_smooth - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        
        # Get color for this joint
        color = np.array(colors[j], dtype=np.float32)
        
        # Create smooth color contribution
        color_contribution = hm_smooth[:, :, None] * color.reshape(1, 1, 3)
        
        # Add to merged image with smooth blending
        merged_rgb += color_contribution * 0.6  # Control overall intensity
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if rgb_image is not None:
        # Overlay on original image
        # Resize merged heatmap to match image dimensions
        H_img, W_img = rgb_image.shape[:2]
        merged_resized = cv2.resize(merged_rgb, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        
        # Normalize the merged heatmap to [0, 1] range
        merged_normalized = np.clip(merged_resized, 0.0, 1.0)
        
        # Create overlay by blending with original image
        # Convert rgb_image to float for proper blending
        img_float = rgb_image.astype(np.float32) / 255.0
        
        # Create smooth overlay with additive blending for more natural look
        overlay = img_float.copy()
        
        # Add the heatmap colors smoothly to the original image
        for c in range(3):
            # Use soft blending - add heatmap colors to original image
            overlay[:, :, c] = img_float[:, :, c] + merged_normalized[:, :, c] * 0.4
        
        # Ensure values are in [0, 1] range
        overlay = np.clip(overlay, 0.0, 1.0)
        
        # Display the overlaid image
        ax.imshow(overlay)
        ax.set_title('Body Pose Features - Overlay on Image', fontsize=16, fontweight='bold', pad=20)
    else:
        # Display the colorful heatmap alone
        ax.imshow(merged_rgb, interpolation='bilinear')
        ax.set_title('Body Pose Features', fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes for clean look
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved Body Pose Features heatmap to {save_path}")
    
    plt.show()
    return merged_rgb

def create_alpha_blended_heatmap(hm_probs: np.ndarray, save_path: str = None):
    """
    Create a blended heatmap where all joints are visible with different transparency levels.
    """
    J, Hm, Wm = hm_probs.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Method 1: Simple sum with normalization
    ax1 = fig.add_subplot(221)
    summed_hm = np.sum(hm_probs, axis=0)
    summed_hm = summed_hm / np.max(summed_hm)  # Normalize
    im1 = ax1.imshow(summed_hm, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax1.set_title('Sum of All Heatmaps (Normalized)', fontsize=14)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Method 2: Maximum across joints
    ax2 = fig.add_subplot(222)
    max_hm = np.max(hm_probs, axis=0)
    im2 = ax2.imshow(max_hm, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax2.set_title('Maximum Across All Joints', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Method 3: Weighted sum (emphasize high-confidence joints)
    ax3 = fig.add_subplot(223)
    weights = np.arange(1, J+1)  # Different weights for each joint
    weighted_sum = np.sum(hm_probs * weights[:, np.newaxis, np.newaxis], axis=0)
    weighted_sum = weighted_sum / np.max(weighted_sum)  # Normalize
    im3 = ax3.imshow(weighted_sum, cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax3.set_title('Weighted Sum (Joint Index)', fontsize=14)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Method 4: Multi-channel visualization (RGB channels)
    ax4 = fig.add_subplot(224)
    # Use first 3 joints for RGB channels, blend others
    r_channel = hm_probs[0]  # Neck
    g_channel = np.mean(hm_probs[1:6], axis=0)  # Right arm joints
    b_channel = np.mean(hm_probs[7:11], axis=0)  # Right leg joints
    
    rgb_blend = np.stack([r_channel, g_channel, b_channel], axis=-1)
    rgb_blend = rgb_blend / np.max(rgb_blend)  # Normalize
    ax4.imshow(rgb_blend, interpolation='nearest')
    ax4.set_title('RGB Blend (Neck, Right Arm, Right Leg)', fontsize=14)
    ax4.axis('off')
    
    plt.suptitle('Different Methods to Merge All 15 Joint Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved blended heatmap comparison to {save_path}")
    
    plt.show()

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

def main():
    parser = argparse.ArgumentParser(description="Visualize pre-generated heatmaps")
    parser.add_argument("--heatmap", type=str, default='/data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/binchen/office//heatmap128/heatmap_000052.npy',
                       help="Path to heatmap .npy file")
    parser.add_argument("--image", type=str, default='/data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/binchen/office/imgs/img_000052.jpg',
                       help="Path to corresponding image (optional, for overlays)")
    parser.add_argument("--save_dir", type=str, default='./overlay_out',
                       help="Where to save visualizations")
    parser.add_argument("--save_overlays", action="store_true", 
                       help="Save per-joint heatmap overlays on the RGB")
    parser.add_argument("--save_joint_points", action="store_true", 
                       help="Save images with joint points marked")
    parser.add_argument("--alpha", type=float, default=0.45, 
                       help="Overlay alpha for heatmaps")
    parser.add_argument("--point_size", type=int, default=8, 
                       help="Size of joint points")
    parser.add_argument("--show_grid", action="store_true", default=True,
                       help="Show grid visualization of all heatmaps")
    parser.add_argument("--show_merged", action="store_true", default=True,
                       help="Show merged heatmap with all joints")
    parser.add_argument("--show_blended", action="store_true", default=False,
                       help="Show different blending methods")
    parser.add_argument("--show_body_pose_features", action="store_true", default=False,
                       help="Show Body Pose Features style heatmap")
    parser.add_argument("--use_different_colors", action="store_true", default=True,
                       help="Use different colors for each joint in merged view")
    
    args = parser.parse_args()
    
    # Load heatmaps
    heatmaps = load_heatmap(args.heatmap)
    
    # Show grid visualization
    if args.show_grid:
        grid_out = os.path.join(args.save_dir, "loaded_heatmaps_grid.png")
        grid_visualization(heatmaps, grid_out)
    
    # Show merged heatmap
    if args.show_merged:
        merged_out = os.path.join(args.save_dir, "merged_heatmap_all_joints.png")
        create_merged_heatmap(heatmaps, merged_out, use_different_colors=args.use_different_colors)
    
    # Show blended heatmap comparison
    if args.show_blended:
        blended_out = os.path.join(args.save_dir, "blended_heatmap_comparison.png")
        create_alpha_blended_heatmap(heatmaps, blended_out)
    
    # Load image if provided for overlays
    rgb = None
    orig_hw = None
    if args.image:
        rgb, orig_hw = load_image(args.image)
    
    # Show Body Pose Features style heatmap
    if args.show_body_pose_features:
        body_pose_out = os.path.join(args.save_dir, "body_pose_features_heatmap.png")
        create_body_pose_features_heatmap(heatmaps, rgb, body_pose_out)
    
    # Save overlays if requested
    if args.save_overlays and rgb is not None:
        save_overlays(rgb, heatmaps, args.save_dir, alpha=args.alpha)
    
    # Save joint points if requested
    if args.save_joint_points and rgb is not None:
        coords, conf = save_joint_points(rgb, heatmaps, args.save_dir, point_size=args.point_size)
        
        # Print joint coordinates and confidences
        print("\nJoint coordinates and confidences:")
        for j, (xy, c) in enumerate(zip(coords, conf)):
            print(f"{j:02d} {JOINT_NAMES[j]:15s}: (x={xy[0]:.1f}, y={xy[1]:.1f}), conf={c:.3f}")
    
    if rgb is None:
        print("No image provided. Use --image to enable overlays and joint point visualization.")

if __name__ == "__main__":
    main()
