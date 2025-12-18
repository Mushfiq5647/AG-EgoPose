import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_heatmap import HeatMap_Network
from options.train_options import TrainOptions

def load_heatmap_model(model_path, device):
    """Load trained heatmap model"""
    opt = TrainOptions().parse()
    model = HeatMap_Network(opt, model_name='resnet18').to(device)
    
    if os.path.exists(model_path):
        print(f"Loading heatmap model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

def preprocess_image(image_path, crop_size=224):
    """Preprocess image for heatmap network"""
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size), antialias=True),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Also create the cropped image for visualization
    cropped_image = image.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    
    return image_tensor, image, cropped_image

def visualize_heatmaps(original_image, cropped_image, heatmaps, save_path=None, joint_names=None):
    """Visualize heatmaps overlaid on both original and cropped images"""
    if joint_names is None:
        joint_names = [f"Joint_{i}" for i in range(heatmaps.shape[0])]
    
    # Resize heatmaps to match both image sizes
    h_orig, w_orig = np.array(original_image).shape[:2]
    h_crop, w_crop = np.array(cropped_image).shape[:2]
    
    heatmaps_resized_orig = F.interpolate(
        torch.from_numpy(heatmaps).unsqueeze(0), 
        size=(h_orig, w_orig), 
        mode='bilinear', 
        align_corners=True
    ).squeeze(0).numpy()
    
    heatmaps_resized_crop = F.interpolate(
        torch.from_numpy(heatmaps).unsqueeze(0), 
        size=(h_crop, w_crop), 
        mode='bilinear', 
        align_corners=True
    ).squeeze(0).numpy()
    
    # Create visualizations
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 10))
    fig2, axes2 = plt.subplots(3, 5, figsize=(20, 12))
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. Combined heatmap overlaid on both images
    for img_idx, (image, heatmaps_resized, title) in enumerate([
        (original_image, heatmaps_resized_orig, "Original Image"),
        (cropped_image, heatmaps_resized_crop, "Cropped Input (224x224)")
    ]):
        axes1[img_idx].imshow(image)
        axes1[img_idx].set_title(f"{title} - Combined Heatmaps", fontsize=16)
        axes1[img_idx].axis('off')
        
        # Combine all heatmaps
        combined_heatmap = np.max(heatmaps_resized, axis=0)
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
        
        # Create colored combined heatmap
        combined_colored = plt.cm.jet(combined_heatmap)
        
        # Overlay combined heatmap on image
        overlay_combined = 0.7 * np.array(image).astype(np.float32) / 255.0 + 0.3 * combined_colored[:, :, :3]
        axes1[img_idx].imshow(overlay_combined)
        
        # Find and plot all peak locations
        colors = plt.cm.tab20(np.linspace(0, 1, len(joint_names)))
        for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
            heatmap = heatmaps_resized[i]
            peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[peak_y, peak_x]
            axes1[img_idx].plot(peak_x, peak_y, 'o', color=color, markersize=8, markeredgewidth=2, 
                              markeredgecolor='white', label=f"{joint_name}: {confidence:.3f}")
        
        axes1[img_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. Individual joint heatmaps overlaid on cropped image (what model sees)
    for i in range(len(joint_names)):
        row = i // 5
        col = i % 5
        
        # Get heatmap for this joint
        heatmap = heatmaps_resized_crop[i]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create colored heatmap
        heatmap_colored = plt.cm.jet(heatmap)
        
        # Overlay on cropped image
        overlay = 0.7 * np.array(cropped_image).astype(np.float32) / 255.0 + 0.3 * heatmap_colored[:, :, :3]
        
        axes2[row, col].imshow(overlay)
        axes2[row, col].set_title(f"{joint_names[i]}", fontsize=10)
        axes2[row, col].axis('off')
        
        # Find and plot peak location
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[peak_y, peak_x]
        axes2[row, col].plot(peak_x, peak_y, 'rx', markersize=8, markeredgewidth=2)
        axes2[row, col].text(5, 15, f"Conf: {confidence:.3f}", color='white', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # 3. Combined heatmap only (without image) for both sizes
    for img_idx, (heatmaps_resized, title) in enumerate([
        (heatmaps_resized_orig, "Original Size"),
        (heatmaps_resized_crop, "Cropped Size (224x224)")
    ]):
        combined_heatmap = np.max(heatmaps_resized, axis=0)
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
        
        axes3[img_idx].imshow(combined_heatmap, cmap='jet')
        axes3[img_idx].set_title(f"Combined Heatmap - {title}", fontsize=16)
        axes3[img_idx].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(axes3[img_idx].images[0], ax=axes3[img_idx], fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', rotation=270, labelpad=15)
        
        # Find and plot all peak locations on combined heatmap
        colors = plt.cm.tab20(np.linspace(0, 1, len(joint_names)))
        for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
            heatmap = heatmaps_resized[i]
            peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[peak_y, peak_x]
            axes3[img_idx].plot(peak_x, peak_y, 'o', color=color, markersize=8, markeredgewidth=2, 
                              markeredgecolor='white', label=f"{joint_name}: {confidence:.3f}")
        
        axes3[img_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        # Save all visualizations
        base_name = save_path.split('.')[0]
        fig1.savefig(f"{base_name}_combined_overlay_comparison.png", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{base_name}_individual_cropped.png", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{base_name}_combined_only_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Heatmap visualizations saved to:")
        print(f"  {base_name}_combined_overlay_comparison.png")
        print(f"  {base_name}_individual_cropped.png")
        print(f"  {base_name}_combined_only_comparison.png")
    
    plt.show()
    fig1.show()
    fig2.show()
    fig3.show()

def test_heatmap_model(model_path, image_path, device='cuda'):
    """Test heatmap model on a single image"""
    # Load model
    model = load_heatmap_model(model_path, device)
    
    # Preprocess image
    image_tensor, original_image, cropped_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    print(f"Input image shape: {image_tensor.shape}")
    print(f"Original image size: {original_image.size}")
    print(f"Cropped image size: {cropped_image.size}")
    
    # Get predictions
    with torch.no_grad():
        heatmaps = model(image_tensor)  # [1, 2*J, H, W]
        
        # Average the two stacks if stereo output
        if heatmaps.shape[1] == 30:  # 2 * 15 joints
            heatmaps = heatmaps.view(1, 2, 15, heatmaps.shape[2], heatmaps.shape[3])
            heatmaps = heatmaps.mean(dim=1)  # Average stereo channels
        elif heatmaps.shape[1] == 15:
            pass  # Already single channel
        else:
            print(f"Unexpected heatmap shape: {heatmaps.shape}")
            return
        
        print(f"Output heatmap shape: {heatmaps.shape}")
        
        # Convert to numpy
        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()  # [J, H, W]
    
    # Define joint names based on your hierarchy
    joint_names = [
        "Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
        "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
        "Left_knee", "Left_ankle", "Left_foot"
    ]
    
    # Visualize
    save_path = f"heatmap_test_{os.path.basename(image_path).split('.')[0]}.png"
    visualize_heatmaps(original_image, cropped_image, heatmaps_np, save_path, joint_names)
    
    # Print peak locations for both original and cropped
    print("\n" + "="*60)
    print("PEAK LOCATIONS AND CONFIDENCE SCORES")
    print("="*60)
    
    # For cropped image (what model actually sees)
    h_crop, w_crop = np.array(cropped_image).shape[:2]
    heatmaps_resized_crop = F.interpolate(
        torch.from_numpy(heatmaps_np).unsqueeze(0), 
        size=(h_crop, w_crop), 
        mode='bilinear', 
        align_corners=True
    ).squeeze(0).numpy()
    
    print(f"\n📊 CROPPED IMAGE ({w_crop}x{h_crop}) - What Model Sees:")
    print("-" * 50)
    for i, joint_name in enumerate(joint_names):
        heatmap = heatmaps_resized_crop[i]
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[peak_y, peak_x]
        print(f"{joint_name:15s}: ({peak_x:3d}, {peak_y:3d}) - Confidence: {confidence:.4f}")
    
    # For original image
    h_orig, w_orig = np.array(original_image).shape[:2]
    heatmaps_resized_orig = F.interpolate(
        torch.from_numpy(heatmaps_np).unsqueeze(0), 
        size=(h_orig, w_orig), 
        mode='bilinear', 
        align_corners=True
    ).squeeze(0).numpy()
    
    print(f"\n📊 ORIGINAL IMAGE ({w_orig}x{h_orig}) - Resized Back:")
    print("-" * 50)
    for i, joint_name in enumerate(joint_names):
        heatmap = heatmaps_resized_orig[i]
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[peak_y, peak_x]
        print(f"{joint_name:15s}: ({peak_x:3d}, {peak_y:3d}) - Confidence: {confidence:.4f}")
    
    # Raw heatmap values (before resizing)
    print(f"\n📊 RAW HEATMAP VALUES ({heatmaps_np.shape[1]}x{heatmaps_np.shape[2]}):")
    print("-" * 50)
    for i, joint_name in enumerate(joint_names):
        heatmap = heatmaps_np[i]
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[peak_y, peak_x]
        print(f"{joint_name:15s}: ({peak_x:3d}, {peak_y:3d}) - Confidence: {confidence:.4f}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    print("• Model was trained on 224x224 cropped images")
    print("• Low confidence scores suggest training issues")
    print("• Same peak locations indicate poor joint differentiation")
    print("• Expected confidence > 0.5 for well-trained models")

def main():
    parser = argparse.ArgumentParser(description='Test heatmap model visualization')
    parser.add_argument('--model_path', type=str, default='/data/My_Backup/ag-egopose/utils/trained_heatmaps/heatmap-020.ckpt',
                       help='Path to trained heatmap model')
    parser.add_argument('--image_path', type=str, default='/data/My_Backup/Dataset/EgoGTA/EgoGTAImages/2020-05-21-13-54-43/img/000190.png',
                       help='Path to test image')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Test the model
    test_heatmap_model(args.model_path, args.image_path, args.device)

if __name__ == '__main__':
    main()
