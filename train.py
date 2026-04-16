# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.amp import autocast, GradScaler

from utils.action_recognition import initialize_actionformer
from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from utils.cross_attention_model import HeatmapToJointFeatures
from utils.cross_attention_model import SpatialJointTransformer
from utils.cross_attention_model import PoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.loss import LossFuncLimb, LossFuncMPJPE, LossFuncCosSim  # Add bone length loss import
from utils.model import  FeatureEncoder
import socket
print("Running on host:", socket.gethostname())
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)
# confirm the integer index as well
if torch.cuda.is_available():
    print("Current CUDA device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# # Number of joints and coordinates per joint
num_joints = 15
coords_per_joint = 3
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

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight,
            a=0.1,  # Better for LeakyReLU
            nonlinearity='leaky_relu'
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def initialize_gelu_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # More conservative, standard for transformers
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# heatmap_embedding: ReLU - use ReLU initialization
def initialize_relu_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight,
            a=0,  # For ReLU
            nonlinearity='relu'
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def check_nan(tensor, name, epoch, i):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}! Epoch: {epoch}, Iteration: {i}")
        return True
    return False

def get_grad_norm(model, model_name=""):
    """Get gradient norm for a model"""
    total_norm = 0
    count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            if param_norm > 0:
                count += 1
            total_norm += param_norm ** 2
    result = total_norm ** (1. / 2)
    if model_name and count == 0:
        print(f"  WARNING: {model_name} has no parameters with non-zero gradients ({sum(1 for p in model.parameters() if p.requires_grad)} total trainable params)")
    return result


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
    loss_log_path = 'loss_log_sceneego_bce.txt'
    with open(loss_log_path, 'w') as f:
        f.write("Epoch, Train_MPJPE, Train_Cos, Train_Bone, Train_Total, Best_Total \n")
    
    # Track best model
    best_loss = float('inf')
    best_epoch = 0

    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size),antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # with open(args.annotation_path, 'rb') as f:
    #     annotation = pickle.load(f)
    opt = TrainOptions().parse()
    data_loader = dataloader_full(opt, transform, mode='train')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    net_heatmap = HeatMap_Network(opt, model_name=args.heatmap_backbone).to(device)
    
    # Load pre-trained 2D heatmap network
    if os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        # Freeze the heatmap network
        for param in net_heatmap.parameters():
            param.requires_grad = False
        net_heatmap.eval()  # Set to evaluation mode after freezing
        print("2D heatmap network loaded and frozen")
        print(f"Using heatmap model: {args.heatmap_trained_path}")
        
        # Debug: Test heatmap output range with a dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256,256).to(device)
            dummy_output = net_heatmap(dummy_input)
            print(f"Heatmap model output range: {dummy_output.min().item():.4f} to {dummy_output.max().item():.4f}")
            print(f"Heatmap model output mean/std: {dummy_output.mean().item():.4f}/{dummy_output.std().item():.4f}")
    else:
        print(f"Warning: Pre-trained heatmap model not found at {args.heatmap_trained_path}")
        print("Will use ground truth heatmaps for training")
    
    # Spatial path: conv encoder extracts rich 128-dim features from 64x64 heatmaps
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=args.hm_embed_dim, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    print("Models initialized")

    # --- ActionFormer selective unfreeze ---
    # Pretrained ActionFormer params are already frozen in initialize_actionformer().
    # Now selectively unfreeze components that need to adapt:
    #   1. input_proj: new layer (DINOv2 384 -> ActionFormer 256), must train
    #   2. embedding convs: must adapt from EgoVLP to DINOv2-projected features
    #   3. last 2 branch transformer blocks: fine-tune temporal representations
    #   4. channel_projector: new layer (multi-scale fusion), must train
    af_unfrozen_params = []
    backbone = actionformer_feature_extractor.model.backbone

    # input_proj is new (randomly initialized), always trainable
    for p in actionformer_feature_extractor.input_proj.parameters():
        p.requires_grad = True
        af_unfrozen_params.append(p)

    # Embedding convs need to adapt to DINOv2-projected input statistics
    for p in backbone.embd.parameters():
        p.requires_grad = True
        af_unfrozen_params.append(p)
    for p in backbone.embd_norm.parameters():
        p.requires_grad = True
        af_unfrozen_params.append(p)

    # Last 2 branch transformer blocks
    num_branch = len(backbone.branch)
    print(f"ActionFormer has {num_branch} branch blocks, unfreezing last 2")
    for blk in backbone.branch[num_branch - 2:]:
        for p in blk.parameters():
            p.requires_grad = True
            af_unfrozen_params.append(p)

    # channel_projector is new (randomly initialized), always trainable
    for p in actionformer_feature_extractor.channel_projector.parameters():
        p.requires_grad = True
        af_unfrozen_params.append(p)

    print(f"ActionFormer unfrozen params: {sum(p.numel() for p in af_unfrozen_params):,}")

    #Initialize weights
    pose_decoder.apply(initialize_gelu_weights)
    spatial_joint_transformer.apply(initialize_gelu_weights)
    heatmap_embedding.apply(initialize_relu_weights)
    #define loss functions
    limb_loss_func = LossFuncLimb().to(device)
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim().to(device)

    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("encoder trainable params:", enc_params)

    # Separate param groups: ActionFormer unfrozen layers get 10x lower LR
    af_param_ids = {id(p) for p in af_unfrozen_params}
    all_other_params = [p for p in (
            list(encoder.parameters()) +
            list(heatmap_embedding.parameters()) +
            list(pose_decoder.parameters()) +
            list(spatial_joint_transformer.parameters())
    ) if p.requires_grad and id(p) not in af_param_ids]

    # Collect all trainable params for gradient clipping
    all_trainable_params = all_other_params + af_unfrozen_params

    optimizer = torch.optim.AdamW([
        {'params': all_other_params, 'lr': args.learning_rate},
        {'params': af_unfrozen_params, 'lr': args.learning_rate * 0.1}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=1e-6
        )
    
    # Mixed precision training for memory efficiency
    scaler = GradScaler(device='cuda')
    # Save checkpoints under --model_path (so experiments are reproducible without editing code)
    model_dir = os.path.abspath(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Checkpoint directory: {model_dir}")
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
            H_hm, W_hm = 64,64
            gt_egoposes = batch['gt_local_pose'].to(device)
            # Compute heatmaps from frozen net_heatmap, then detach and enable gradients
            # This allows: net_heatmap (frozen) → heatmaps (differentiable inputs) → heatmap_embedding.encoder (trainable)
            with torch.no_grad():
                all_images_flat = images.view(-1, 3, H_img, W_img)  # [B*T, 3, H, W]
                all_heatmaps = net_heatmap(all_images_flat)  # [B*T, J, H_hm, W_hm]
                all_heatmaps = torch.sigmoid(all_heatmaps)  # Convert logits to 0-1 range
            # Detach from frozen net_heatmap, but enable gradients for downstream modules
            heatmaps = all_heatmaps.detach().view(B, T, 15, H_hm, W_hm)  # [B, T, 15, H_hm, W_hm]
            heatmaps.requires_grad_(True)  # Enable gradient computation for heatmap_embedding/spatial_transformer
            
            # Only encoder (heavy DINOv2 + ActionFormer) needs autocast for memory efficiency
            with autocast(device_type='cuda'):
                motion_features = encoder(images)                       # (B, T, 384)

            # All cross-stream modules in float32 to prevent gradient underflow
            # (these are lightweight — autocast savings are negligible but float16 kills their gradients)
            motion_features = motion_features.float()

            # Spatial path: conv encoder → rich 128-dim features
            heatmap_features = heatmap_embedding(heatmaps)              # (B,T,J,128)
            spatial_joint_features = spatial_joint_transformer(heatmap_features)  # (B,T,J,128)

            if i == 0:
                print(f"DEBUG: heatmap_features shape={heatmap_features.shape}, dtype={heatmap_features.dtype}")
                print(f"DEBUG: spatial_joint_features shape={spatial_joint_features.shape}")

            # Pose decoder fuses spatial + temporal features directly
            with autocast(device_type='cuda'):
                pose_logits = pose_decoder(spatial_joint_features, motion_features)
                # Reshape to pose format and convert to FP32 for loss computation
                final = pose_logits.view(B, T, num_joints, 3).float()  # Convert to FP32

                # Reshape for loss computation
                final_reshaped = final.reshape(B * T, num_joints, 3)
                gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)

                # Compute losses (all in FP32)
                mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
                bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
                cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)

                # Combined loss (no regularization for now - trajectories have very small gradients)
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

            # Backward and optimize with AMP
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()

            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, args.clip_value)
            
            # Optional: Monitor gradient norms for debugging
            if i % args.log_step == 0:
                print(f'Encoder grad norm: {get_grad_norm(encoder, "Encoder"):.6f}')
                print(f'Decoder grad norm: {get_grad_norm(pose_decoder, "Decoder"):.6f}')
                print(f'Heatmap grad norm: {get_grad_norm(heatmap_embedding, "Heatmap"):.6f}')
                print(f'Spatial grad norm: {get_grad_norm(spatial_joint_transformer, "Spatial"):.6f}')
                
                # Monitor learning rate
                current_lr = optimizer.param_groups[0]['lr']
            
            # Step optimizer with scaler
            scaler.step(optimizer)
            scaler.update()
            
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},'
                      .format(epoch, args.num_epochs, i, total_step, final_loss.item()))
                # torch.save(decoder.state_dict(), os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                # torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

            # Note: Model saving is now handled at the epoch level (best model + periodic checkpoints)

        # Calculate training epoch averages
        train_avg_mpjpe = np.mean(train_mpjpe_losses)
        train_avg_cos = np.mean(train_cos_losses)
        train_avg_bone = np.mean(train_bone_losses)
        train_avg_total = np.mean(train_total_losses)

        # Check if this is the best model so far
        if train_avg_total < best_loss:
            best_loss = train_avg_total
            best_epoch = epoch
            
            # Save best model checkpoints
            print(f"🎯 New best model! Loss: {best_loss:.4f} (Epoch {best_epoch})")
            torch.save(pose_decoder.state_dict(),
                       os.path.join(model_dir, 'pose-decoder-best.ckpt'))
            torch.save(encoder.state_dict(),
                       os.path.join(model_dir, 'encoder-best.ckpt'))
            torch.save(heatmap_embedding.state_dict(),
                       os.path.join(model_dir, 'heatmap_embedding-best.ckpt'))
            torch.save(spatial_joint_transformer.state_dict(),
                       os.path.join(model_dir, 'spatial_transformer-best.ckpt'))
            print(f"Saved best checkpoints to {model_dir}")

        # Save periodic checkpoints
        if (epoch + 1) % args.save_interval == 0:
            print(f"Saving periodic checkpoints for epoch {epoch + 1} to {model_dir}")
            torch.save(pose_decoder.state_dict(),
                       os.path.join(model_dir, f'pose-decoder-{epoch + 1:03d}.ckpt'))
            torch.save(encoder.state_dict(),
                       os.path.join(model_dir, f'encoder-{epoch + 1:03d}.ckpt'))
            torch.save(heatmap_embedding.state_dict(),
                       os.path.join(model_dir, f'heatmap_embedding-{epoch + 1:03d}.ckpt'))
            torch.save(spatial_joint_transformer.state_dict(),
                       os.path.join(model_dir, f'spatial_transformer-{epoch + 1:03d}.ckpt'))
            print(f"Saved periodic checkpoints for epoch {epoch + 1} to {model_dir}")

        # Print epoch summary
        print(f"=== Epoch {epoch} Summary ===")
        print(
            f"Train - MPJPE: {train_avg_mpjpe:.4f}, Cos: {train_avg_cos:.4f}, Bone: {train_avg_bone:.4f}, Total: {train_avg_total:.4f}")
        print(f"Best loss so far: {best_loss:.4f} (Epoch {best_epoch})")

        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print("New learning rate:", new_lr)
        with open(loss_log_path, 'a') as log_f:
            log_f.write(
                f"{epoch},"
                f"{train_avg_mpjpe:.4f},{train_avg_cos:.4f},"
                f"{train_avg_bone:.4f},{train_avg_total:.4f},{best_loss:.4f}\n")

    # Save final model state
    print(f"\n🏁 Training completed!")
    print(f"Best model was from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Best model checkpoints saved as:")
    print(f"  • pose-decoder-best.ckpt")
    print(f"  • encoder-best.ckpt")
    print(f"  • heatmap_embedding-best.ckpt")
    print(f"  • spatial_transformer-best.ckpt")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    # parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')
    parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
    parser.add_argument('--heatmap_backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='backbone for 2D heatmap network (default: convnext_tiny)')

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
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--seq_length', type=int, default=64, help='length of the pose/video sequences')
    parser.add_argument('--stride', type=int, default=64, help='sliding window stride for dataset construction')
    parser.add_argument('--crop_size', type=int, default=256, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lambda_heatmap', type=float, default=0.1, help='weight for heatmap loss')

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')
    parser.add_argument('--save_interval', type=int, default=2, help='save checkpoint every N epochs')

    args = parser.parse_args()
    print(args)
    main(args)
