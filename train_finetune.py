# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

from action_recognition import initialize_actionformer
from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from utils.cross_attention_model import HeatmapToJointFeatures
from utils.cross_attention_model import SpatialJointTransformer
from heatmaps.network_heatmap import HeatMap_Network
from utils.loss import LossFuncLimb, LossFuncMPJPE, LossFuncCosSim
from utils.model import FeatureEncoder, PoseDecoder
import socket

print("Running on host:", socket.gethostname())
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)
# confirm the integer index as well
print("Current CUDA device index:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(device.index))

# Number of joints and coordinates per joint
num_joints = 15
# coords_per_jo    int = 3
#
# # Define bidirectional connections for the central body part
# bidirectional_connections = [
#     (0, 1),  # SpineBase <-> SpineMid
#     (1, 20),  # SpineMid <-> SpineShoulder
#     (20, 2),  # SpineShoulder <-> Neck
#     (2, 3),  # Neck <-> Head
#     (0, 12),  # SpineBase <-> HipLeft
#     (0, 16)  # SpineBase <-> HipRight
# ]
#
# unidirectional_connections = [
#     # Left arm chain
#     (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),
#     # Right arm chain
#     (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),
#     # Left leg chain
#     (12, 13), (13, 14), (14, 15),
#     # Right leg chain
#     (16, 17), (17, 18), (18, 19)
# ]
#
# edge_index = [[], []]
#
# for joint_a, joint_b in bidirectional_connections:
#     for j in range(coords_per_joint):
#         edge_index[0].append(joint_a * coords_per_joint + j)
#         edge_index[1].append(joint_b * coords_per_joint + j)
#         edge_index[0].append(joint_b * coords_per_joint + j)
#         edge_index[1].append(joint_a * coords_per_joint + j)
#
# # Add unidirectional edges (one direction for each pair)
# for joint_a, joint_b in unidirectional_connections:
#     for j in range(coords_per_joint):
#         edge_index[0].append(joint_a * coords_per_joint + j)
#         edge_index[1].append(joint_b * coords_per_joint + j)
#
# # Optionally, add self-loops for each joint
# for joint in range(num_joints):
#     for j in range(coords_per_joint):
#         edge_index[0].append(joint * coords_per_joint + j)
#         edge_index[1].append(joint * coords_per_joint + j)
#
# edge_index = torch.tensor(edge_index, dtype=torch.long)

# Note: initialize_weights function removed for fine-tuning
# We load pre-trained weights instead of initializing from scratch

def check_nan(tensor, name, epoch, i):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}! Epoch: {epoch}, Iteration: {i}")
        return True
    return False

def get_grad_norm(model):
    """Get gradient norm for a model"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def update_learning_rate(self):
    old_lr = self.optimizers[0].param_groups[0]['lr']
    for scheduler in self.schedulers:
        scheduler.step()
    lr = self.optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

def main(args):
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Create loss log file
    loss_log_path = 'loss_log_finetune.txt'
    with open(loss_log_path, 'w') as f:
        f.write("Epoch, Train_MPJPE, Train_Cos, Train_Bone, Train_Total \n")

    # image preprocessing
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(args.crop_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    opt = TrainOptions().parse()
    data_loader = dataloader_full(opt, transform, mode='train')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    
    # Initialize models
    net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=128, feature_dim=128, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    pose_decoder = PoseDecoder(motion_dim=384, joint_dim=128, out_dim=3).to(device)
    
    # Load pre-trained 2D heatmap network
    if os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        net_heatmap.eval()  # Set to evaluation mode
        # Freeze the heatmap network
        for param in net_heatmap.parameters():
            param.requires_grad = False
        print("2D heatmap network loaded and frozen")
    else:
        print(f"Warning: Pre-trained heatmap model not found at {args.heatmap_trained_path}")
        print("Will use ground truth heatmaps for training")
    
    # Load pre-trained models for fine-tuning
    print("Loading pre-trained models for fine-tuning...")
    
    # Load encoder
    if os.path.exists(args.encoder_path):
        print(f"Loading encoder from {args.encoder_path}")
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        print("Encoder loaded successfully")
    else:
        print(f"Warning: Encoder not found at {args.encoder_path}")
    
    # Load pose decoder
    if os.path.exists(args.decoder_path):
        print(f"Loading pose decoder from {args.decoder_path}")
        pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        print("Pose decoder loaded successfully")
    else:
        print(f"Warning: Pose decoder not found at {args.decoder_path}")
    
    # Load heatmap embedding
    if os.path.exists(args.heatmap_path):
        print(f"Loading heatmap embedding from {args.heatmap_path}")
        heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))
        print("Heatmap embedding loaded successfully")
    else:
        print(f"Warning: Heatmap embedding not found at {args.heatmap_path}")
    
    # Load spatial joint transformer
    if os.path.exists(args.spatial_transformer_path):
        print(f"Loading spatial joint transformer from {args.spatial_transformer_path}")
        spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))
        print("Spatial joint transformer loaded successfully")
    else:
        print(f"Warning: Spatial joint transformer not found at {args.spatial_transformer_path}")
    
    print("All models initialized and loaded")
    
    # loss and optimizer
    criterion = nn.MSELoss()
    limb_loss_func = LossFuncLimb().to(device)  # Use existing bone length loss
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim().to(device)
    params = (
            list(encoder.parameters()) +
            list(heatmap_embedding.parameters()) +
            list(pose_decoder.parameters()) +
            list(spatial_joint_transformer.parameters())
    )

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=5e-4
        )
    # No mixed precision - using full precision training
    model_dir = os.path.abspath('./utils/sceneego')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    total_step = len(data_loader)
    
    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
        encoder.train()
        pose_decoder.train()
        heatmap_embedding.train()
        spatial_joint_transformer.train()
        
        # Initialize training loss accumulators
        train_mpjpe_losses = []
        train_cos_losses = []
        train_bone_losses = []
        train_total_losses = []

        for i, (batch) in enumerate(data_loader):
            print("Printing iteration number:", i)
            # Instead of: for i, (images, homography, gt_egoposes, lengths) in enumerate(data_loader)
            images = batch['input_rgb'].to(device)  # Tensor
            B, T, _, H_img, W_img = images.shape
            H_hm, W_hm = 128,128
            gt_egoposes = batch['gt_local_pose'].to(device)
            gt_egoposes = gt_egoposes
            
            # MEMORY OPTIMIZATION: Process heatmaps in smaller chunks
            chunk_size = 16  # Process 16 frames at a time instead of 64
            all_heatmaps = []
            
            with torch.no_grad():
                for t in range(0, T, chunk_size):
                    end_t = min(t + chunk_size, T)
                    img_chunk = images[:, t:end_t].contiguous()  # [B, chunk_size, 3, H, W]
                    img_chunk_flat = img_chunk.view(-1, 3, H_img, W_img)  # [B*chunk_size, 3, H, W]
                    
                    hm_chunk = net_heatmap(img_chunk_flat)  # [B*chunk_size, J, H_hm, W_hm]
                    hm_chunk = hm_chunk.view(B, end_t - t, 15, H_hm, W_hm)
                    all_heatmaps.append(hm_chunk)
                    
            heatmaps = torch.cat(all_heatmaps, dim=1)  # [B, T, 15, H_hm, W_hm]
            
            # Forward pass without autocast for full precision
            motion_features = encoder(images)
            heatmap_features = heatmap_embedding(heatmaps)
            spatial_joint_features = spatial_joint_transformer(heatmap_features)
            pose_logits = pose_decoder(spatial_joint_features, motion_features)
            
            # Reshape to pose format
            final = pose_logits.view(B, T, num_joints, 3)
            
            # Reshape for loss computation
            final_reshaped = final.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)
            
            # Compute losses
            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
            cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)
            
            # Combined loss
            final_loss = (
                         opt.lambda_mpjpe * mpjpe_loss + 
                         opt.lambda_cos_sim * cos_loss + 
                         opt.lambda_bone_length * bone_length_loss)
            
            # Store losses for epoch averaging
            train_mpjpe_losses.append(mpjpe_loss.item())
            train_cos_losses.append(cos_loss.item())
            train_bone_losses.append(bone_length_loss.item())
            train_total_losses.append(final_loss.item())
            
            print(f"MPJPE: {mpjpe_loss:.4f}, "
                  f"Cos: {cos_loss.item():.4f}, Bone: {bone_length_loss.item():.4f}")

            # Backward and optimize without mixed precision
            optimizer.zero_grad()
            final_loss.backward()
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_value)
            
            # Optional: Monitor gradient norms for debugging
            if i % args.log_step == 0:
                print(f'Total gradient norm: {total_norm:.4f}')
                print(f'Encoder grad norm: {get_grad_norm(encoder):.4f}')
                print(f'Decoder grad norm: {get_grad_norm(pose_decoder):.4f}')
                print(f'Heatmap grad norm: {get_grad_norm(heatmap_embedding):.4f}')
                print(f'Spatial grad norm: {get_grad_norm(spatial_joint_transformer):.4f}')
                
                # Monitor learning rate
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_lr:.6f}')
            
            # Step optimizer
            optimizer.step()
            
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, final_loss.item(), np.exp(final_loss.item())))

            if i == total_step-1:
                print('Saving model...')
                torch.save(pose_decoder.state_dict(),
                           os.path.join(model_dir, 'pose-decoder-finetune-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(model_dir, 'encoder-finetune-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(heatmap_embedding.state_dict(),
                           os.path.join(model_dir, 'heatmap_embedding-finetune-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(spatial_joint_transformer.state_dict(),
                           os.path.join(model_dir, 'spatial_transformer-finetune-{}-{}.ckpt'.format(epoch + 1, i + 1)))

        # Calculate training epoch averages
        train_avg_mpjpe = np.mean(train_mpjpe_losses)
        train_avg_cos = np.mean(train_cos_losses)
        train_avg_bone = np.mean(train_bone_losses)
        train_avg_total = np.mean(train_total_losses)

        # Print epoch summary
        print(f"=== Epoch {epoch} Summary ===")
        print(
            f"Train - MPJPE: {train_avg_mpjpe:.4f}, Cos: {train_avg_cos:.4f}, Bone: {train_avg_bone:.4f}, Total: {train_avg_total:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print("New learning rate:", new_lr)
        with open(loss_log_path, 'a') as log_f:
            log_f.write(
                f"{epoch},"
                f"{train_avg_mpjpe:.4f},{train_avg_cos:.4f},"
                f"{train_avg_bone:.4f},{train_avg_total:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--model_path', type=str, default='./utils/trained_finetuned_sceneego', help='path for saving fine-tuned models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')
    parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
    
    # Pre-trained model paths for fine-tuning
    parser.add_argument('--encoder_path', type=str, default='./utils/trained_egopwfull_mo2cap2/encoder-040.ckpt', help='path for pre-trained encoder')
    parser.add_argument('--decoder_path', type=str, default= './utils/trained_egopwfull_mo2cap2/pose-decoder-040.ckpt', help='path for pre-trained decoder')
    parser.add_argument('--heatmap_path', type=str, default='./utils/trained_egopwfull_mo2cap2/heatmap_embedding-040.ckpt',help = 'path for pre-trained heatmap embedding')
    parser.add_argument('--spatial_transformer_path', type=str, default = './utils/trained_egopwfull_mo2cap2/spatial_transformer-040.ckpt', help='path for pre-trained spatial transformer')
    parser.add_argument('--image_dir', type=str, default='/data/My_Backup/UnrealEgo/scripts/data/UnrealEgoData',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='D:/Dataset/EgoPW_dataset/EgoPW_dataset_release',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')

    parser.add_argument('--embed_feature_dim', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hm_embed_dim', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=0.001)  # Lower learning rate for fine-tuning
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=64, help='length of the pose/video sequences')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=20)  # Fewer epochs for fine-tuning
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lambda_heatmap', type=float, default=0.1, help='weight for heatmap loss')

    parser.add_argument('--log_step', type=int, default=20, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=20, help='step size for saving trained models')

    args = parser.parse_args()
    print(args)
    main(args)
