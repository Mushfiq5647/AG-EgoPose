import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from PIL import Image
import pickle

from action_recognition import initialize_actionformer
from options.test_options import TestOptions
from utils.cross_attention_model import HeatmapToJointFeatures, SpatialJointTransformer, PoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.util import batch_compute_similarity_transform_torch
from utils.loss import LossFuncMPJPE


torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_joints = 15

print("Using device:", device)

# Joint connections for skeleton visualization (15 joints)
SKELETON_CONNECTIONS = [
    (0, 1),   # Neck to Right_shoulder
    (0, 4),   # Neck to Left_shoulder
    (1, 2),   # Right_shoulder to Right_elbow
    (2, 3),   # Right_elbow to Right_wrist
    (4, 5),   # Left_shoulder to Left_elbow
    (5, 6),   # Left_elbow to Left_wrist
    (1, 7),   # Right_shoulder to Right_hip
    (4, 11),  # Left_shoulder to Left_hip
    (7, 8),   # Right_hip to Right_knee
    (8, 9),   # Right_knee to Right_ankle
    (9, 10),  # Right_ankle to Right_foot
    (11, 12), # Left_hip to Left_knee
    (12, 13), # Left_knee to Left_ankle
    (13, 14), # Left_ankle to Left_foot
    (7, 11),  # Right_hip to Left_hip
]

JOINT_NAMES = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist",
    "Left_shoulder", "Left_elbow", "Left_wrist", "Right_hip",
    "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
    "Left_knee", "Left_ankle", "Left_foot"
]

def plot_3d_pose(poses, title="3D Pose", connections=None, ax=None, color='darkblue', alpha=0.7, line_color=None):
    """Plot 3D pose with skeleton connections"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Use line_color if provided, otherwise use joint color
    if line_color is None:
        line_color = color
    
    # Plot joints (thicker markers)
    ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c=color, s=160, alpha=alpha)
    
    # Plot skeleton connections
    if connections is not None:
        for start_idx, end_idx in connections:
            if start_idx < len(poses) and end_idx < len(poses):
                start_point = poses[start_idx]
                end_point = poses[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       color=line_color, linewidth=4, alpha=alpha)
    
    # Remove all matplotlib elements for plain white background
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Set equal aspect ratio
    max_range = np.array([poses[:,0].max()-poses[:,0].min(),
                         poses[:,1].max()-poses[:,1].min(),
                         poses[:,2].max()-poses[:,2].min()]).max() / 2.0
    mid_x = (poses[:,0].max()+poses[:,0].min()) * 0.5
    mid_y = (poses[:,1].max()+poses[:,1].min()) * 0.5
    mid_z = (poses[:,2].max()+poses[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax

def calculate_joint_errors(gt_poses, pred_poses):
    """Calculate per-joint errors"""
    errors = np.linalg.norm(gt_poses - pred_poses, axis=-1)  # (T, J)
    return errors

def main(args):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Load and preprocess the input image
    print(f"Loading image from {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # For temporal models, we need to create a sequence
    # Duplicate the single image to create a sequence of length seq_length
    images = image_tensor.unsqueeze(1).repeat(1, args.seq_length, 1, 1, 1)  # [1, T, 3, H, W]
    
    # Initialize ActionFormer
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    
    # Freeze ActionFormer
    for param in actionformer_feature_extractor.parameters():
        param.requires_grad = False
    
    # Load options using TestOptions (same as visualize_3d_poses.py)
    opt = TestOptions().parse()
    
    net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=128, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    
    # Load pre-trained models
    print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
    net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
    net_heatmap.eval()
    for param in net_heatmap.parameters():
        param.requires_grad = False
    
    print(f"Loading encoder from {args.encoder_path}")
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    
    print(f"Loading pose decoder from {args.decoder_path}")
    pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    
    print(f"Loading heatmap embedding from {args.heatmap_path}")
    heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))
    
    print(f"Loading spatial transformer from {args.spatial_transformer_path}")
    spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))
    
    # Set all models to evaluation mode
    net_heatmap.eval()
    encoder.eval()
    pose_decoder.eval()
    heatmap_embedding.eval()
    spatial_joint_transformer.eval()
    
    print("All models loaded and set to evaluation mode")
    
    print("Running pose estimation on the input image...")
    
    with torch.no_grad():
        B, T, _, H_img, W_img = images.shape
        
        # Process the image
        H_hm, W_hm = 64, 64
        all_images_flat = images.view(-1, 3, H_img, W_img)  # [T, 3, H, W]
        all_heatmaps = net_heatmap(all_images_flat)  # [T, J, H_hm, W_hm]
        all_heatmaps = torch.sigmoid(all_heatmaps)  # Convert logits to 0-1 range
        heatmaps = all_heatmaps.view(B, T, 15, H_hm, W_hm)  # [1, T, 15, H_hm, W_hm]
        
        motion_features = encoder(images)
        heatmap_features = heatmap_embedding(heatmaps)
        spatial_joint_features = spatial_joint_transformer(heatmap_features)
        pose_logits = pose_decoder(heatmap_features, motion_features)

        # Reshape to pose format
        final = pose_logits.view(B, T, num_joints, 3)
        
        # Since we duplicated the single image T times, all frames have the same input
        # Use frame 0 for the prediction (all frames should give similar predictions)
        frame_idx = 0
        pred_pose_tensor = final[0, frame_idx:frame_idx+1]  # [1, 15, 3]
        
        print(f"Predicted 3D pose shape: {pred_pose_tensor.shape}")
        print(f"Note: Input image was duplicated {T} times for temporal model, using frame {frame_idx}")
        
        # Load ground truth from pickle file if provided
        gt_pose = None
        if args.gt_pickle_path is not None and os.path.exists(args.gt_pickle_path):
            # Extract image filename from full path
            image_filename = os.path.basename(args.image_path)
            print(f"Loading pickle file: {args.gt_pickle_path}")
            print(f"Searching for image: {image_filename}")
            
            with open(args.gt_pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # Find the item where 'image_name' matches
            for item in data:
                if item.get('image_name') == image_filename:
                    gt_pose = item.get('ego_pose_gt')
                    print(f"Found ground truth for {image_filename}")
                    print(f"Ground truth shape: {gt_pose.shape}")
                    break
            
            if gt_pose is None:
                print(f"Warning: Image {image_filename} not found in pickle file")
        
        # If ground truth is available, apply Procrustes alignment
        if gt_pose is not None:
            gt_pose_tensor = torch.from_numpy(gt_pose).unsqueeze(0).float().to(device)  # [1, 15, 3]
            
            # Apply Procrustes alignment
            aligned_pose_tensor = batch_compute_similarity_transform_torch(pred_pose_tensor, gt_pose_tensor)
            aligned_pose = aligned_pose_tensor[0].cpu().numpy()
            gt_pose_np = gt_pose
            
            # Calculate errors
            mpjpe_before = torch.sqrt(((pred_pose_tensor - gt_pose_tensor) ** 2).sum(dim=-1)).mean().item()
            mpjpe_after = torch.sqrt(((aligned_pose_tensor - gt_pose_tensor) ** 2).sum(dim=-1)).mean().item()
            
            print(f"MPJPE (before alignment): {mpjpe_before:.2f} mm")
            print(f"PA-MPJPE (after Procrustes): {mpjpe_after:.2f} mm")
            
            # Visualize GT and Procrustes-aligned prediction superimposed
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Plot ground truth (red)
            plot_3d_pose(gt_pose_np, "", SKELETON_CONNECTIONS, ax, 
                        color='red', alpha=0.8, line_color='darkred')
            
            # Plot Procrustes-aligned prediction (blue) on the same axes
            plot_3d_pose(aligned_pose, "", SKELETON_CONNECTIONS, ax, 
                        color='darkblue', alpha=0.8, line_color='#28A428')
            
            # Adjust axis limits to fit both poses
            all_poses = np.concatenate([gt_pose_np, aligned_pose], axis=0)
            max_range = np.array([all_poses[:,0].max()-all_poses[:,0].min(),
                                 all_poses[:,1].max()-all_poses[:,1].min(),
                                 all_poses[:,2].max()-all_poses[:,2].min()]).max() / 2.0
            mid_x = (all_poses[:,0].max()+all_poses[:,0].min()) * 0.5
            mid_y = (all_poses[:,1].max()+all_poses[:,1].min()) * 0.5
            mid_z = (all_poses[:,2].max()+all_poses[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                       markersize=10, label='Ground Truth', alpha=0.8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
                       markersize=10, label='Procrustes-Aligned Prediction', alpha=0.8)
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.show()
        else:
            # No ground truth - just show raw prediction
            print("No ground truth provided - showing raw prediction only")
            pred_pose = pred_pose_tensor[0].cpu().numpy()
            
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Plot predicted pose - blue joints with green lines
            plot_3d_pose(pred_pose, "", 
                        SKELETON_CONNECTIONS, ax, color='darkblue', alpha=0.8, line_color='#32CD32')
            
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input image
    parser.add_argument('--image_path', type=str, default ='/data/My_Backup/Dataset/SceneEgo_test/diogo1/imgs/img_002098.jpg')
    
    # Ground truth pickle file for Procrustes alignment
    parser.add_argument('--gt_pickle_path', type=str, default='/data/My_Backup/Dataset/SceneEgo_test/jian2/annotation.pkl', help='Path to pickle file with ground truth poses')
    
    # Model paths
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
    parser.add_argument('--heatmap_path', type=str, required=True, help='path for trained heatmap embedding')
    parser.add_argument('--spatial_transformer_path', type=str, required=True, help='path for trained spatial transformer')
    
    # Model parameters
    parser.add_argument('--hm_embed_dim', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--seq_length', type=int, default=64, help='sequence length for temporal model')
    
    args = parser.parse_args()
    main(args)

