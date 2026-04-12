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
from utils.cross_attention_model import SpatialStatsExtractor
from utils.cross_attention_model import SpatialJointTransformer
from utils.cross_attention_model import PerJointTrajectoryTokens
from utils.cross_attention_model import ActionInformedVisibilityEstimation
from utils.cross_attention_model import PoseDecoder
from utils.cross_attention_model import StereoFeatureFusion
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

# # Number of coordinates per joint
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


@torch.no_grad()
def run_validation(val_loader, net_heatmap, encoder, heatmap_embedding,
                   spatial_stats_extractor, stereo_fusion, spatial_joint_transformer,
                   pjtt, aive, pose_decoder,
                   mpjpe_loss_func, limb_loss_func, cos_sim_loss_func, opt,
                   num_heatmap_joints, num_pose_joints):
    """Run a full validation pass on the stereo pipeline. Returns dict of mean losses."""
    encoder.eval(); heatmap_embedding.eval(); spatial_joint_transformer.eval()
    stereo_fusion.eval(); pjtt.eval(); aive.eval(); pose_decoder.eval()

    mpjpe_list, cos_list, bone_list, total_list = [], [], [], []
    H_hm = W_hm = 64

    for batch in val_loader:
        images_left = batch['input_rgb_left'].to(device)
        images_right = batch['input_rgb_right'].to(device)
        gt_egoposes = batch['gt_local_pose'].to(device) / 100.0
        B, T, _, H_img, W_img = images_left.shape

        left_flat = images_left.reshape(-1, 3, H_img, W_img)
        right_flat = images_right.reshape(-1, 3, H_img, W_img)
        stereo_logits = net_heatmap(left_flat, right_flat)
        stereo_probs = torch.sigmoid(stereo_logits)
        heatmaps_left = stereo_probs[:, :num_heatmap_joints].view(B, T, num_heatmap_joints, H_hm, W_hm)
        heatmaps_right = stereo_probs[:, num_heatmap_joints:].view(B, T, num_heatmap_joints, H_hm, W_hm)

        with autocast(device_type='cuda'):
            motion_features = encoder(images_left, images_right)
        motion_features = motion_features.float()

        hm_feat_left = heatmap_embedding(heatmaps_left)
        hm_feat_right = heatmap_embedding(heatmaps_right)
        heatmap_features = stereo_fusion(hm_feat_left, hm_feat_right)
        spatial_stats = 0.5 * (spatial_stats_extractor(heatmaps_left)
                               + spatial_stats_extractor(heatmaps_right))

        spatial_joint_features = spatial_joint_transformer(heatmap_features)
        traj_tokens = pjtt(motion_features)
        gated_features = aive(spatial_joint_features, spatial_stats, traj_tokens)

        pose_logits = pose_decoder(gated_features, motion_features)
        final = pose_logits.view(B, T, num_pose_joints, 3).float()
        final_reshaped = final.reshape(B * T, num_pose_joints, 3)
        gt_reshaped = gt_egoposes.reshape(B * T, num_pose_joints, 3)

        mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
        bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
        cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)
        total = (opt.lambda_mpjpe * mpjpe_loss
                 + opt.lambda_cos_sim * cos_loss
                 + opt.lambda_bone_length * bone_length_loss)

        mpjpe_list.append(mpjpe_loss.item())
        cos_list.append(cos_loss.item())
        bone_list.append(bone_length_loss.item())
        total_list.append(total.item())

    return {
        'mpjpe': float(np.mean(mpjpe_list)),
        'cos':   float(np.mean(cos_list)),
        'bone':  float(np.mean(bone_list)),
        'total': float(np.mean(total_list)),
    }


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
    loss_log_path = 'loss_log_egopw_bce.txt'
    with open(loss_log_path, 'w') as f:
        f.write("Epoch, Train_MPJPE, Train_Cos, Train_Bone, Train_Total, "
                "Val_MPJPE, Val_Cos, Val_Bone, Val_Total, Best_Total \n")
    
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
    if opt.model == 'unrealego':
        num_heatmap_joints = 15
        num_pose_joints = 16
    else:
        num_heatmap_joints = opt.num_heatmap
        num_pose_joints = opt.num_heatmap
    opt.num_heatmap = num_heatmap_joints
    print(f"Using {num_heatmap_joints} heatmap joints and {num_pose_joints} pose joints for model={opt.model}")
    data_loader = dataloader_full(opt, transform, mode='train')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    try:
        val_loader = dataloader_full(opt, transform, mode='validation')
        print(f"Validation loader: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
    except (FileNotFoundError, OSError) as e:
        print(f"⚠️  No validation set available ({e}). Will track best by train loss.")
        val_loader = None
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
        
        # Debug: Test heatmap output range with dummy stereo inputs
        with torch.no_grad():
            dummy_left = torch.randn(1, 3, 256, 256).to(device)
            dummy_right = torch.randn(1, 3, 256, 256).to(device)
            dummy_output = net_heatmap(dummy_left, dummy_right)
            print(f"Heatmap model output shape: {tuple(dummy_output.shape)}  (expects num_heatmap*2 channels)")
            print(f"Heatmap model output range: {dummy_output.min().item():.4f} to {dummy_output.max().item():.4f}")
            print(f"Heatmap model output mean/std: {dummy_output.mean().item():.4f}/{dummy_output.std().item():.4f}")
    else:
        print(f"Warning: Pre-trained heatmap model not found at {args.heatmap_trained_path}")
        print("Will use ground truth heatmaps for training")
    
    # Main spatial path: conv encoder extracts rich 128-dim features from 64x64 heatmaps
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=args.hm_embed_dim, method='conv_pool').to(device)
    # Side path: lightweight stats (8-dim) for AIVE visibility gate only (no learnable params)
    spatial_stats_extractor = SpatialStatsExtractor(heatmap_size=64).to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(
        args.hm_embed_dim, num_heads=4, num_layers=3, num_joints=num_heatmap_joints
    ).to(device)
    # Stereo fusion: gated concat+project over mean baseline for per-joint features
    stereo_fusion = StereoFeatureFusion(dim=args.hm_embed_dim).to(device)
    pjtt = PerJointTrajectoryTokens(num_joints=num_heatmap_joints, motion_dim=384, num_heads=4).to(device)
    aive = ActionInformedVisibilityEstimation(motion_dim=384, joint_dim=args.hm_embed_dim).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3, num_joints=num_pose_joints).to(device)
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
    pjtt.apply(initialize_gelu_weights)
    aive.apply(initialize_relu_weights)
    stereo_fusion.apply(initialize_gelu_weights)

    #define loss functions
    limb_loss_func = LossFuncLimb(num_joints=num_pose_joints).to(device)
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim(num_joints=num_pose_joints).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("encoder trainable params:", enc_params)

    # Separate param groups: ActionFormer unfrozen layers get 10x lower LR
    af_param_ids = {id(p) for p in af_unfrozen_params}
    all_other_params = [p for p in (
            list(encoder.parameters()) +
            list(heatmap_embedding.parameters()) +
            list(pose_decoder.parameters()) +
            list(spatial_joint_transformer.parameters()) +
            list(stereo_fusion.parameters()) +
            list(pjtt.parameters()) +
            list(aive.parameters())
    ) if p.requires_grad and id(p) not in af_param_ids]

    # DEBUG: Check if PJTT and AIVE are in optimizer
    pjtt_params_set = {id(p) for p in pjtt.parameters()}
    aive_params_set = {id(p) for p in aive.parameters()}
    other_params_set = {id(p) for p in all_other_params}

    pjtt_in_optimizer = len(pjtt_params_set & other_params_set)
    aive_in_optimizer = len(aive_params_set & other_params_set)
    print(f"DEBUG: PJTT params in optimizer: {pjtt_in_optimizer}/{len(list(pjtt.parameters()))}")
    print(f"DEBUG: AIVE params in optimizer: {aive_in_optimizer}/{len(list(aive.parameters()))}")

    # Collect all trainable params for gradient clipping
    all_trainable_params = all_other_params + af_unfrozen_params

    optimizer = torch.optim.AdamW([
        {'params': all_other_params, 'lr': args.learning_rate},
        {'params': af_unfrozen_params, 'lr': args.learning_rate * 0.1}
    ], weight_decay=args.weight_decay)

    # DEBUG: Verify PJTT and AIVE are in optimizer
    opt_param_ids = set()
    for param_group in optimizer.param_groups:
        opt_param_ids.update(id(p) for p in param_group['params'])

    pjtt_in_opt = sum(1 for p in pjtt.parameters() if id(p) in opt_param_ids)
    aive_in_opt = sum(1 for p in aive.parameters() if id(p) in opt_param_ids)
    print(f"DEBUG: PJTT in optimizer: {pjtt_in_opt}/{sum(1 for _ in pjtt.parameters())} params")
    print(f"DEBUG: AIVE in optimizer: {aive_in_opt}/{sum(1 for _ in aive.parameters())} params")

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
        stereo_fusion.train()
        pjtt.train()
        aive.train()
        # Initialize training loss accumulators
        train_mpjpe_losses = []
        train_cos_losses = []
        train_bone_losses = []
        train_total_losses = []

        for i, (batch) in enumerate(data_loader):
            print("Printing iteration number:", i)
            # Stereo inputs from UnrealEgo dataloader
            images_left = batch['input_rgb_left'].to(device)
            images_right = batch['input_rgb_right'].to(device)
            B, T, _, H_img, W_img = images_left.shape
            H_hm, W_hm = 64, 64
            gt_egoposes = batch['gt_local_pose'].to(device) / 100.0

            # Compute stereo heatmaps from frozen net_heatmap.
            # Network does internal channel-wise stereo fusion and outputs num_heatmap*2 channels:
            # first num_heatmap = left, last num_heatmap = right.
            with torch.no_grad():
                left_flat = images_left.reshape(-1, 3, H_img, W_img)    # [B*T, 3, H, W]
                right_flat = images_right.reshape(-1, 3, H_img, W_img)  # [B*T, 3, H, W]
                stereo_logits = net_heatmap(left_flat, right_flat)      # [B*T, J*2, H_hm, W_hm]
                stereo_probs = torch.sigmoid(stereo_logits)
                left_hm_flat = stereo_probs[:, :num_heatmap_joints]
                right_hm_flat = stereo_probs[:, num_heatmap_joints:]

            heatmaps_left = left_hm_flat.detach().view(B, T, num_heatmap_joints, H_hm, W_hm)
            heatmaps_right = right_hm_flat.detach().view(B, T, num_heatmap_joints, H_hm, W_hm)
            heatmaps_left.requires_grad_(True)
            heatmaps_right.requires_grad_(True)

            # Stereo encoder (heavy DINOv2 + ActionFormer): autocast for memory.
            # Encoder uses symmetric mean+|diff| fusion on left/right CLS features
            # before ActionFormer, preserving cross-view disagreement cheaply.
            with autocast(device_type='cuda'):
                motion_features = encoder(images_left, images_right)    # (B, T, 384)

            # All cross-stream modules in float32 to prevent gradient underflow
            # (these are lightweight — autocast savings are negligible but float16 kills their gradients)
            motion_features = motion_features.float()

            # Spatial path with shared-weight per-view encoding + stereo fusion.
            # heatmap_embedding (conv_pool) is shared across views — runs on each independently.
            hm_feat_left = heatmap_embedding(heatmaps_left)             # (B,T,J,128)
            hm_feat_right = heatmap_embedding(heatmaps_right)           # (B,T,J,128)
            heatmap_features = stereo_fusion(hm_feat_left, hm_feat_right)  # (B,T,J,128)

            # Stats path: average left/right uncertainty stats for AIVE gate.
            # Symmetric mean is appropriate here — stats are interpretable scalars (peak,
            # entropy, coords) and we want a view-invariant uncertainty estimate.
            stats_left = spatial_stats_extractor(heatmaps_left)         # (B,T,J,8)
            stats_right = spatial_stats_extractor(heatmaps_right)       # (B,T,J,8)
            spatial_stats = 0.5 * (stats_left + stats_right)

            spatial_joint_features = spatial_joint_transformer(heatmap_features)  # (B,T,J,128)
            traj_tokens = pjtt(motion_features)                     # (B, T, J, 384)
            gated_features = aive(spatial_joint_features, spatial_stats, traj_tokens)  # (B,T,J,128)

            if i == 0:
                print(f"DEBUG: heatmap_features shape={heatmap_features.shape}, dtype={heatmap_features.dtype}")
                print(f"DEBUG: spatial_stats shape={spatial_stats.shape}")
                print(f"DEBUG: traj_tokens shape={traj_tokens.shape}")
                print(f"DEBUG: gated_features shape={gated_features.shape}")

            # Only pose decoder needs autocast (it's the largest trainable module)
            with autocast(device_type='cuda'):
                pose_logits = pose_decoder(gated_features, motion_features)
                # Reshape to pose format and convert to FP32 for loss computation
                final = pose_logits.view(B, T, num_pose_joints, 3).float()  # Convert to FP32

                # Reshape for loss computation
                final_reshaped = final.reshape(B * T, num_pose_joints, 3)
                gt_reshaped = gt_egoposes.reshape(B * T, num_pose_joints, 3)

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

            # DEBUG: Check if PJTT and AIVE got gradients
            if i % args.log_step == 0:
                pjtt_grads = [(name, p.grad.abs().max().item() if p.grad is not None else 0.0) for name, p in pjtt.named_parameters() if p.requires_grad]
                aive_grads = [(name, p.grad.abs().max().item() if p.grad is not None else 0.0) for name, p in aive.named_parameters() if p.requires_grad]
                pjtt_nonzero = sum(1 for _, v in pjtt_grads if v > 0)
                aive_nonzero = sum(1 for _, v in aive_grads if v > 0)
                print(f"DEBUG: PJTT {pjtt_nonzero}/{len(pjtt_grads)} params have non-zero grads")
                if pjtt_grads:
                    print(f"  → {', '.join(f'{n}:{v:.2e}' for n, v in pjtt_grads[:3])}")
                print(f"DEBUG: AIVE {aive_nonzero}/{len(aive_grads)} params have non-zero grads")
                if aive_grads:
                    print(f"  → {', '.join(f'{n}:{v:.2e}' for n, v in aive_grads[:3])}")

            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, args.clip_value)
            
            # Optional: Monitor gradient norms for debugging
            if i % args.log_step == 0:
                print(f'Encoder grad norm: {get_grad_norm(encoder, "Encoder"):.6f}')
                print(f'Decoder grad norm: {get_grad_norm(pose_decoder, "Decoder"):.6f}')
                print(f'Heatmap grad norm: {get_grad_norm(heatmap_embedding, "Heatmap"):.6f}')
                print(f'Spatial grad norm: {get_grad_norm(spatial_joint_transformer, "Spatial"):.6f}')
                print(f'PJTT grad norm: {get_grad_norm(pjtt, "PJTT"):.2e}')
                print(f'AIVE grad norm: {get_grad_norm(aive, "AIVE"):.2e}')
                
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

        # Run validation after every training epoch
        val_metrics = None
        if val_loader is not None:
            print(f"Running validation for epoch {epoch}...")
            val_metrics = run_validation(
                val_loader, net_heatmap, encoder, heatmap_embedding,
                spatial_stats_extractor, stereo_fusion, spatial_joint_transformer,
                pjtt, aive, pose_decoder,
                mpjpe_loss_func, limb_loss_func, cos_sim_loss_func, opt,
                num_heatmap_joints, num_pose_joints,
            )
            print(f"Val   - MPJPE: {val_metrics['mpjpe']:.4f}, Cos: {val_metrics['cos']:.4f}, "
                  f"Bone: {val_metrics['bone']:.4f}, Total: {val_metrics['total']:.4f}")

        # Track best model: prefer validation loss if available, otherwise train loss
        tracked_loss = val_metrics['total'] if val_metrics is not None else train_avg_total
        if tracked_loss < best_loss:
            best_loss = tracked_loss
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
            torch.save(stereo_fusion.state_dict(),
                       os.path.join(model_dir, 'stereo_fusion-best.ckpt'))
            torch.save(pjtt.state_dict(),
                       os.path.join(model_dir, 'pjtt-best.ckpt'))
            torch.save(aive.state_dict(),
                       os.path.join(model_dir, 'aive-best.ckpt'))
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
            torch.save(stereo_fusion.state_dict(),
                       os.path.join(model_dir, f'stereo_fusion-{epoch + 1:03d}.ckpt'))
            torch.save(pjtt.state_dict(),
                       os.path.join(model_dir, f'pjtt-{epoch + 1:03d}.ckpt'))
            torch.save(aive.state_dict(),
                       os.path.join(model_dir, f'aive-{epoch + 1:03d}.ckpt'))
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
            if val_metrics is not None:
                log_f.write(
                    f"{epoch},"
                    f"{train_avg_mpjpe:.4f},{train_avg_cos:.4f},"
                    f"{train_avg_bone:.4f},{train_avg_total:.4f},"
                    f"{val_metrics['mpjpe']:.4f},{val_metrics['cos']:.4f},"
                    f"{val_metrics['bone']:.4f},{val_metrics['total']:.4f},"
                    f"{best_loss:.4f}\n")
            else:
                log_f.write(
                    f"{epoch},"
                    f"{train_avg_mpjpe:.4f},{train_avg_cos:.4f},"
                    f"{train_avg_bone:.4f},{train_avg_total:.4f},"
                    f"NA,NA,NA,NA,{best_loss:.4f}\n")

    # Save final model state
    print(f"\n🏁 Training completed!")
    print(f"Best model was from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Best model checkpoints saved as:")
    print(f"  • pose-decoder-best.ckpt")
    print(f"  • encoder-best.ckpt")
    print(f"  • heatmap_embedding-best.ckpt")
    print(f"  • spatial_transformer-best.ckpt")
    print(f"  • pjtt-best.ckpt")
    print(f"  • aive-best.ckpt")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')
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

    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lambda_heatmap', type=float, default=0.1, help='weight for heatmap loss')

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')
    parser.add_argument('--save_interval', type=int, default=2, help='save checkpoint every N epochs')

    args = parser.parse_args()
    print(args)
    main(args)
