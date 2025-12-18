"""
Create "Body Pose Features" visualization
Overlays feature maps on the original image with colorful gradients
"""
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.action_recognition import initialize_actionformer
from scipy.ndimage import gaussian_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path: str, img_size: int, device: str):
    """Load and preprocess image"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    
    # Keep original for visualization
    original_rgb = rgb.copy()
    
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    x = preprocess(rgb).unsqueeze(0).to(device)
    return x, original_rgb, (H0, W0)


def create_colorful_gradient_colormap():
    """Create a colormap similar to the reference image: teal -> yellow -> pink"""
    colors = [
        (0.0, 0.4, 0.5),   # Dark teal/cyan (low values)
        (0.2, 0.6, 0.6),   # Lighter teal
        (0.5, 0.8, 0.5),   # Greenish
        (0.9, 0.9, 0.3),   # Yellow
        (1.0, 0.7, 0.5),   # Orange/peach
        (0.9, 0.6, 0.7),   # Pink (high values)
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('body_features', colors, N=n_bins)
    return cmap


def extract_resnet_features(encoder, images):
    """Extract spatial feature maps from ResNet-50 layer4"""
    feature_maps = {}
    
    def hook_fn(module, input, output):
        feature_maps['layer4'] = output
    
    # Register hook on ResNet layer4 (last conv layer before pooling)
    hook = encoder.resnet[-2].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        images_transposed = images.transpose(0, 1)
        for batch in images_transposed:
            _ = encoder.resnet(batch)
            break  # Only first frame
    
    hook.remove()
    
    # Get feature maps: [B, C, H, W]
    feat = feature_maps['layer4'][0]  # [C, H, W]
    return feat


def visualize_body_pose_features(rgb_image, feature_map, save_path, sigma=5.0, alpha=0.6):
    """
    Create "Body Pose Features" visualization
    
    Args:
        rgb_image: Original RGB image (H, W, 3)
        feature_map: Feature map from ResNet (C, H_feat, W_feat)
        save_path: Where to save the visualization
        sigma: Gaussian blur sigma for smoothing
        alpha: Transparency of overlay (0-1)
    """
    H_orig, W_orig = rgb_image.shape[:2]
    
    # Average across channels and normalize
    feat_avg = feature_map.mean(dim=0).cpu().numpy()  # [H_feat, W_feat]
    feat_avg = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min() + 1e-8)
    
    # Smooth the features
    feat_avg = gaussian_filter(feat_avg, sigma=sigma)
    
    # Resize to match original image size
    feat_resized = cv2.resize(feat_avg, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    cmap = create_colorful_gradient_colormap()
    colored_features = cmap(feat_resized)[:, :, :3]  # RGB, drop alpha
    colored_features = (colored_features * 255).astype(np.uint8)
    
    # Blend with original image
    rgb_normalized = rgb_image.astype(np.float32)
    colored_normalized = colored_features.astype(np.float32)
    
    blended = cv2.addWeighted(rgb_normalized, 1.0 - alpha, colored_normalized, alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Body Pose Features Visualization", fontsize=16, fontweight='bold')
    
    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Feature map only
    axes[1].imshow(feat_resized, cmap=cmap)
    axes[1].set_title("Feature Map", fontsize=12)
    axes[1].axis('off')
    
    # Blended result
    axes[2].imshow(blended)
    axes[2].set_title("Body Pose Features", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved body pose features visualization to {save_path}")
    plt.close()
    
    # Save just the blended result for paper
    paper_path = save_path.replace('.png', '_paper.png')
    cv2.imwrite(paper_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved paper figure to {paper_path}")
    
    return blended


def visualize_heatmap_embedding_features(rgb_image, heatmap_embedding, heatmaps, save_path, sigma=3.0, alpha=0.7):
    """
    Create "Body Pose Features" visualization from HeatmapToJointFeatures
    Shows features extracted by the heatmap embedding network
    """
    H_orig, W_orig = rgb_image.shape[:2]
    
    # Extract intermediate feature maps from HeatmapToJointFeatures conv layers
    feature_maps = {}
    
    def hook_conv1(module, input, output):
        feature_maps['conv1'] = output  # [B*T*J, 16, 64, 64]
    
    def hook_conv2(module, input, output):
        feature_maps['conv2'] = output  # [B*T*J, 32, 32, 32]
    
    def hook_conv3(module, input, output):
        feature_maps['conv3'] = output  # [B*T*J, 64, 16, 16]
    
    # Register hooks on conv layers
    hook1 = heatmap_embedding.conv_layers[0].register_forward_hook(hook_conv1)
    hook2 = heatmap_embedding.conv_layers[2].register_forward_hook(hook_conv2)
    hook3 = heatmap_embedding.conv_layers[4].register_forward_hook(hook_conv3)
    
    with torch.no_grad():
        _ = heatmap_embedding(heatmaps)
    
    hook1.remove()
    hook2.remove()
    hook3.remove()
    
    # Use conv1 features (16 channels, 64x64) - first layer after input
    # Average across all joints and channels to get a single feature map
    conv1_feat = feature_maps['conv1']  # [B*T*J, 16, 64, 64]
    B, T, J, H, W = heatmaps.shape
    conv1_feat = conv1_feat.view(B, T, J, 16, 64, 64)
    
    # Average across joints and channels for first frame
    feature_map = conv1_feat[0, 0].mean(dim=0).cpu().numpy()  # Average across joints -> [16, 64, 64] -> mean -> [64, 64]
    feature_map = feature_map.mean(axis=0)  # Average across channels -> [64, 64]
    
    # Smooth
    feature_map = gaussian_filter(feature_map, sigma=sigma)
    
    # Normalize
    feature_map = (feature_map - feature_map.min()) / \
                  (feature_map.max() - feature_map.min() + 1e-8)
    
    # Resize to original image size
    feature_map_resized = cv2.resize(feature_map, (W_orig, H_orig), 
                                      interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    cmap = create_colorful_gradient_colormap()
    colored_features = cmap(feature_map_resized)[:, :, :3]  # RGB
    colored_features = (colored_features * 255).astype(np.uint8)
    
    # Blend with original image
    blended = cv2.addWeighted(rgb_image.astype(np.float32), 1.0 - alpha,
                             colored_features.astype(np.float32), alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Body Pose Features from HeatmapToJointFeatures", fontsize=16, fontweight='bold')
    
    # Original
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Feature map only
    axes[1].imshow(feature_map_resized, cmap=cmap)
    axes[1].set_title("Heatmap Embedding Features (Conv1)", fontsize=12)
    axes[1].axis('off')
    
    # Blended
    axes[2].imshow(blended)
    axes[2].set_title("Body Pose Features", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved heatmap embedding features visualization to {save_path}")
    plt.close()
    
    # Save for paper
    paper_path = save_path.replace('.png', '_paper.png')
    cv2.imwrite(paper_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved paper figure to {paper_path}")
    
    return blended


def visualize_heatmap_body_features(rgb_image, heatmaps, save_path, sigma=3.0, alpha=0.7):
    """
    Create "Body Pose Features" visualization from combined heatmaps
    Shows all joint locations with colorful gradient overlay
    """
    H_orig, W_orig = rgb_image.shape[:2]
    
    # Combine all joint heatmaps (max across joints)
    combined_heatmap = torch.max(heatmaps[0, 0], dim=0)[0].cpu().numpy()  # [H_hm, W_hm]
    
    # Smooth
    combined_heatmap = gaussian_filter(combined_heatmap, sigma=sigma)
    
    # Normalize
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / \
                       (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
    
    # Resize to original image size
    heatmap_resized = cv2.resize(combined_heatmap, (W_orig, H_orig), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    cmap = create_colorful_gradient_colormap()
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # RGB
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    blended = cv2.addWeighted(rgb_image.astype(np.float32), 1.0 - alpha,
                             colored_heatmap.astype(np.float32), alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Body Pose Features from Heatmaps", fontsize=16, fontweight='bold')
    
    # Original
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap_resized, cmap=cmap)
    axes[1].set_title("Combined Joint Heatmap", fontsize=12)
    axes[1].axis('off')
    
    # Blended
    axes[2].imshow(blended)
    axes[2].set_title("Body Pose Features", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved heatmap-based body features to {save_path}")
    plt.close()
    
    # Save for paper
    paper_path = save_path.replace('.png', '_paper.png')
    cv2.imwrite(paper_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved paper figure to {paper_path}")
    
    return blended


def main():
    parser = argparse.ArgumentParser(description="Create Body Pose Features visualization")
    parser.add_argument("--image", type=str, default='/data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/ayush_new/office1/imgs/img_000072.jpg', help="Path to input image")
    parser.add_argument("--heatmap_ckpt", type=str, default='utils/trained_heatmaps/bce_combined/heatmap_best.ckpt', help="Heatmap model checkpoint")
    parser.add_argument("--heatmap_embed_path", type=str, default='utils/trained_egopwtrain_bce_seq32/heatmap_embedding-best.ckpt',
                       help="Heatmap embedding checkpoint (for HeatmapToJointFeatures)")
    parser.add_argument("--encoder_path", type=str, default='utils/trained_egopwtrain_bce_seq32/encoder-best.ckpt', 
                       help="Encoder checkpoint (only needed for resnet method)")
    parser.add_argument("--config_path", type=str, default='actionformer/config/ego4D_egovlp.yaml')
    parser.add_argument("--save_dir", type=str, default='./body_features', help="Output directory")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay transparency (0-1)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur sigma")
    parser.add_argument("--method", type=str, default='heatmap', choices=['resnet', 'heatmap', 'both'],
                       help="Which features to visualize")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image: {args.image}")
    x, rgb, orig_hw = load_image(args.image, args.img_size, device)
    
    # Duplicate to create sequence
    images = x.unsqueeze(1).repeat(1, args.seq_length, 1, 1, 1)  # [1, T, 3, H, W]
    
    # Resize original RGB to model input size for consistent visualization
    rgb_resized = cv2.resize(rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    
    if args.method in ['resnet', 'both']:
        # Initialize models for ResNet features
        if not args.encoder_path:
            raise ValueError("--encoder_path is required when using --method resnet or both")
        
        print("Initializing models for ResNet features...")
        actionformer = initialize_actionformer(args.config_path)
        for param in actionformer.parameters():
            param.requires_grad = False
        
        encoder = FeatureEncoder(actionformer).to(device)
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        encoder.eval()
        
        # Extract ResNet features
        print("Extracting ResNet features...")
        feature_map = extract_resnet_features(encoder, images)
        
        # Visualize
        print("Creating ResNet-based body pose features...")
        visualize_body_pose_features(
            rgb_resized, feature_map,
            os.path.join(args.save_dir, "body_features_resnet.png"),
            sigma=args.sigma, alpha=args.alpha
        )
    
    if args.method in ['heatmap', 'both']:
        # Initialize heatmap model
        print("Initializing heatmap model...")
        # Create a dummy options object instead of parsing command line args
        class DummyOpt:
            def __init__(self):
                self.num_heatmap = 15
                self.init_ImageNet = True
        opt = DummyOpt()
        net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
        net_heatmap.load_state_dict(torch.load(args.heatmap_ckpt, map_location=device))
        net_heatmap.eval()
        
        # Initialize heatmap embedding
        from utils.cross_attention_model import HeatmapToJointFeatures
        heatmap_embedding = HeatmapToJointFeatures(
            heatmap_size=64, feature_dim=128, method='conv_pool'
        ).to(device)
        
        # Load heatmap embedding if checkpoint provided
        if args.heatmap_embed_path:
            print(f"Loading heatmap embedding from {args.heatmap_embed_path}")
            heatmap_embedding.load_state_dict(torch.load(args.heatmap_embed_path, map_location=device))
        heatmap_embedding.eval()
        
        # Generate heatmaps
        print("Generating heatmaps...")
        with torch.no_grad():
            B, T, _, H_img, W_img = images.shape
            all_images_flat = images.view(-1, 3, H_img, W_img)
            all_heatmaps = net_heatmap(all_images_flat)
            all_heatmaps = torch.sigmoid(all_heatmaps)
            heatmaps = all_heatmaps.view(B, T, 15, 64, 64)
        
        # Visualize raw heatmaps
        print("Creating heatmap-based body pose features (raw heatmaps)...")
        visualize_heatmap_body_features(
            rgb_resized, heatmaps,
            os.path.join(args.save_dir, "body_features_heatmap_raw.png"),
            sigma=args.sigma * 0.6, alpha=args.alpha
        )
        
        # Visualize heatmap embedding features
        print("Creating body pose features from HeatmapToJointFeatures...")
        visualize_heatmap_embedding_features(
            rgb_resized, heatmap_embedding, heatmaps,
            os.path.join(args.save_dir, "body_features_heatmap_embedding.png"),
            sigma=args.sigma, alpha=args.alpha
        )
    
    print(f"\nAll visualizations saved to {args.save_dir}")
    print("Use the '*_paper.png' versions for your paper figures")


if __name__ == "__main__":
    main()

# # Heatmap-based (recommended - shows joint locations)
# python visualize_body_pose_features.py \
#     --method heatmap \
#     --alpha 0.7 \
#     --sigma 2.0 \
#     --save_dir ./body_features
#
# # ResNet-based (shows general spatial features)
# python visualize_body_pose_features.py \
#     --image /path/to/image.jpg \
#     --encoder_path /path/to/encoder.ckpt \
#     --method resnet \
#     --alpha 0.6 \
#     --sigma 5.0 \
#     --save_dir ./body_features
#
# # Both methods
# python visualize_body_pose_features.py \
#     --image /path/to/image.jpg \
#     --heatmap_ckpt /path/to/heatmap.ckpt \
#     --encoder_path /path/to/encoder.ckpt \
#     --method both \
#     --save_dir ./body_features