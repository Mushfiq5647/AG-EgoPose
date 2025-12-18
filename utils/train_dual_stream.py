#!/usr/bin/env python3
"""
Training script for Dual Stream Pose Estimation Model with Cross Attention
"""

import argparse
import os
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils.action_recognition import initialize_actionformer
from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from heatmaps.network_heatmap import HeatMap_Network
from utils.loss import LossFuncLimb, LossFuncMPJPE, LossFuncCosSim
from utils.model import FeatureEncoder
from utils.cross_attention_model import create_dual_stream_model
import matplotlib.pyplot as plt
import seaborn as sns

print("Running on host:", socket.gethostname())
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print("Current CUDA device index:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(device.index))

# Number of joints
num_joints = 15


def initialize_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def visualize_attention_weights(attention_weights, epoch, save_dir):
    """
    Visualize cross attention weights
    
    Args:
        attention_weights: (B, T, J) tensor
        epoch: Current epoch number
        save_dir: Directory to save plots
    """
    # Average across batch and time
    avg_attention = attention_weights.mean(dim=[0, 1]).cpu().numpy()  # (J,)
    
    # Joint names (customize based on your skeleton)
    joint_names = [
        'Head', 'Neck', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist', 'Torso', 'LHip',
        'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle'
    ]
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(avg_attention)), avg_attention)
    plt.xticks(range(len(joint_names)), joint_names, rotation=45)
    plt.ylabel('Average Attention Weight')
    plt.title(f'Cross Attention Weights - Epoch {epoch}')
    plt.grid(True, alpha=0.3)
    
    # Color bars by attention strength
    colors = plt.cm.viridis(avg_attention / avg_attention.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Heatmap over time (sample first sequence from batch)
    plt.subplot(1, 2, 2)
    time_attention = attention_weights[0].cpu().numpy()  # (T, J)
    sns.heatmap(time_attention.T, 
                xticklabels=False, 
                yticklabels=joint_names,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'})
    plt.xlabel('Time Steps')
    plt.ylabel('Joints')
    plt.title(f'Attention Over Time - Epoch {epoch}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'attention_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Initialize ActionFormer
    print("Initializing ActionFormer...")
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    
    # Create model directories
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    attention_dir = os.path.join(args.model_path, 'attention_visualizations')
    if not os.path.exists(attention_dir):
        os.makedirs(attention_dir)
    
    # Create loss log file
    loss_log_path = os.path.join(args.model_path, 'dual_stream_loss_log.txt')
    with open(loss_log_path, 'w') as f:
        f.write("Epoch,Train_MPJPE,Train_Cos,Train_Bone,Train_Total,Heatmap_Quality\n")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(args.crop_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Data loading
    opt = TrainOptions().parse()
    data_loader = dataloader_full(opt, transform, mode='train')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    
    # Initialize heatmap network
    net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
    
    # Load pre-trained 2D heatmap network if available
    if args.heatmap_trained_path and os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        net_heatmap.eval()
        for param in net_heatmap.parameters():
            param.requires_grad = False
        print("2D heatmap network loaded and frozen")
        use_predicted_heatmaps = True
    else:
        print("No pre-trained heatmap model found. Using ground truth heatmaps.")
        use_predicted_heatmaps = False
    
    # Initialize feature encoder (for ActionFormer input)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    
    # Create dual stream model
    print("Creating dual stream model...")
    model_config = {
        'motion_dim': args.embed_feature_dim,
        'joint_feature_dim': args.joint_feature_dim,
        'num_joints': num_joints,
        'num_heads': args.num_heads,
        'num_transformer_layers': args.num_transformer_layers,
        'heatmap_size': 128,
        'output_pose_dim': 3
    }
    
    dual_stream_model = create_dual_stream_model(
        actionformer_feature_extractor, 
        config=model_config
    ).to(device)
    
    # Apply weight initialization to trainable parts
    dual_stream_model.heatmap_converter.apply(initialize_weights)
    dual_stream_model.joint_transformer.apply(initialize_weights)
    dual_stream_model.cross_attention.apply(initialize_weights)
    dual_stream_model.joint_aggregator.apply(initialize_weights)
    dual_stream_model.pose_decoder.apply(initialize_weights)
    
    encoder.apply(initialize_weights)
    
    # Loss functions
    criterion = nn.MSELoss()
    limb_loss_func = LossFuncLimb().to(device)
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim().to(device)
    
    # Optimizer - only trainable parameters
    trainable_params = (
        list(encoder.parameters()) +
        list(dual_stream_model.heatmap_converter.parameters()) +
        list(dual_stream_model.joint_transformer.parameters()) +
        list(dual_stream_model.cross_attention.parameters()) +
        list(dual_stream_model.joint_aggregator.parameters()) +
        list(dual_stream_model.pose_decoder.parameters())
    )
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=5e-5
    )
    
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")
    
    # Training loop
    total_step = len(data_loader)
    
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
        
        # Set models to training mode
        encoder.train()
        dual_stream_model.train()
        
        # Initialize loss accumulators
        train_mpjpe_losses = []
        train_cos_losses = []
        train_bone_losses = []
        train_total_losses = []
        heatmap_quality_losses = []
        attention_weights_list = []
        
        for i, batch in enumerate(data_loader):
            if i % 10 == 0:  # Print progress every 10 iterations
                print(f"Processing batch {i}/{total_step}")
            
            # Extract data
            images = batch['input_rgb'].to(device)  # (B, T, 3, H, W)
            gt_heatmaps = batch['gt_heatmap'].to(device)  # (B, T, J, H, W)
            gt_poses = batch['gt_local_pose'].to(device)  # (B, T, J, 3)
            
            B, T, _, H_img, W_img = images.shape
            _, _, J, H_hm, W_hm = gt_heatmaps.shape
            
            # Get image features for ActionFormer
            image_features = encoder(images)  # (B, T, embed_feature_dim)
            
            # Get heatmaps (predicted or GT)
            if use_predicted_heatmaps:
                with torch.no_grad():
                    images_flat = images.view(B * T, 3, H_img, W_img)
                    predicted_heatmaps = net_heatmap(images_flat)  # (B*T, J, H_pred, W_pred)
                    # Resize to match GT size
                    predicted_heatmaps = F.interpolate(
                        predicted_heatmaps, size=(H_hm, W_hm), 
                        mode='bilinear', align_corners=True
                    )
                    predicted_heatmaps = predicted_heatmaps.view(B, T, J, H_hm, W_hm)
                
                # Monitor heatmap quality
                heatmap_quality = F.mse_loss(predicted_heatmaps, gt_heatmaps)
                heatmap_quality_losses.append(heatmap_quality.item())
                
                # Use predicted heatmaps
                heatmaps = predicted_heatmaps
            else:
                # Use GT heatmaps
                heatmaps = gt_heatmaps
                heatmap_quality = torch.tensor(0.0, device=device)
                heatmap_quality_losses.append(0.0)
            
            # Forward pass through dual stream model
            predicted_poses, attention_weights, enhanced_joints = dual_stream_model(
                image_features, heatmaps
            )
            
            # Store attention weights for visualization
            if i == 0:  # Store first batch attention for visualization
                attention_weights_list.append(attention_weights.detach())
            
            # Reshape for loss computation
            B, T = predicted_poses.shape[:2]
            pred_reshaped = predicted_poses.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_poses.reshape(B * T, num_joints, 3)
            
            # Compute losses
            mpjpe_loss = mpjpe_loss_func(pred_reshaped, gt_reshaped)
            bone_length_loss = limb_loss_func(pred_reshaped, gt_reshaped)
            cos_loss = cos_sim_loss_func(pred_reshaped, gt_reshaped)
            
            # Combined loss
            final_loss = (
                opt.lambda_mpjpe * mpjpe_loss +
                opt.lambda_cos_sim * cos_loss +
                opt.lambda_bone_length * bone_length_loss
            )
            
            # Store losses
            train_mpjpe_losses.append(mpjpe_loss.item())
            train_cos_losses.append(cos_loss.item())
            train_bone_losses.append(bone_length_loss.item())
            train_total_losses.append(final_loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            final_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_value)
            
            optimizer.step()
            
            # Logging
            if i % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{total_step}]')
                print(f'MPJPE: {mpjpe_loss:.4f}, Cos: {cos_loss:.4f}, Bone: {bone_length_loss:.4f}')
                print(f'Total Loss: {final_loss:.4f}, Heatmap Quality: {heatmap_quality:.6f}')
                
                # Print attention summary
                if len(attention_weights_list) > 0:
                    avg_attention = attention_weights.mean(dim=[0, 1])  # (J,)
                    top_joints = torch.topk(avg_attention, 3).indices.tolist()
                    print(f'Top attended joints: {top_joints}')
        
        # Epoch summary
        train_avg_mpjpe = np.mean(train_mpjpe_losses)
        train_avg_cos = np.mean(train_cos_losses)
        train_avg_bone = np.mean(train_bone_losses)
        train_avg_total = np.mean(train_total_losses)
        train_avg_heatmap = np.mean(heatmap_quality_losses)
        
        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Train - MPJPE: {train_avg_mpjpe:.4f}, Cos: {train_avg_cos:.4f}")
        print(f"Bone: {train_avg_bone:.4f}, Total: {train_avg_total:.4f}")
        print(f"Heatmap Quality: {train_avg_heatmap:.6f}")
        
        # Save attention visualization
        if len(attention_weights_list) > 0 and epoch % args.vis_freq == 0:
            visualize_attention_weights(
                attention_weights_list[0], epoch, attention_dir
            )
        
        # Learning rate update
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save model checkpoint
        if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.model_path, f'dual_stream_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': dual_stream_model.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_avg_mpjpe': train_avg_mpjpe,
                'config': model_config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Log to file
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch+1},{train_avg_mpjpe:.6f},{train_avg_cos:.6f},"
                   f"{train_avg_bone:.6f},{train_avg_total:.6f},{train_avg_heatmap:.6f}\n")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Stream Pose Estimation Training')
    
    # Model paths
    parser.add_argument('--config_path', type=str, 
                       default='actionformer/config/ego4D_egovlp.yaml',
                       help='ActionFormer config file path')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--annotation_path', type=str, required=True,
                       help='Annotation file path')
    parser.add_argument('--heatmap_trained_path', type=str, default=None,
                       help='Pre-trained 2D heatmap model path')
    
    # Data parameters
    parser.add_argument('--crop_size', type=int, default=224,
                       help='Image crop size')
    
    # Model architecture parameters
    parser.add_argument('--embed_feature_dim', type=int, default=384,
                       help='ActionFormer feature dimension')
    parser.add_argument('--joint_feature_dim', type=int, default=128,
                       help='Joint feature dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=3,
                       help='Number of transformer layers')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--clip_value', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging and saving
    parser.add_argument('--log_step', type=int, default=20,
                       help='Print log every N steps')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save model every N epochs')
    parser.add_argument('--vis_freq', type=int, default=5,
                       help='Visualize attention every N epochs')
    
    args = parser.parse_args()
    print("Arguments:", args)
    
    main(args)
