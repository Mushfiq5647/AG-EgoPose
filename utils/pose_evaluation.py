#!/usr/bin/env python3
"""
BA-MPJPE implementation: Bone alignment followed by PA-MPJPE
Based on the baseline paper's approach
"""

import numpy as np
import torch
from copy import deepcopy
from utils.rigid_transform_with_scale import umeyama_parameters


def align_skeleton_size(estimated_seq, gt_seq):
    """
    Align skeleton size using scale-only transformation (from baseline paper)
    
    Args:
        estimated_seq: (N, J, 3) or (B, T, J, 3) estimated poses
        gt_seq: (N, J, 3) or (B, T, J, 3) ground truth poses
        
    Returns:
        np.ndarray: Scale-aligned estimated poses
    """
    estimated_seq = deepcopy(np.asarray(estimated_seq))
    gt_seq = deepcopy(np.asarray(gt_seq))
    
    # Reshape to (N, J, 3) if needed
    if estimated_seq.ndim == 4:
        B, T, J, C = estimated_seq.shape
        estimated_seq = estimated_seq.reshape(-1, J, C)
        gt_seq = gt_seq.reshape(-1, J, C)
    
    N = estimated_seq.shape[0]
    aligned_pose_list = np.zeros_like(estimated_seq)
    
    for i in range(N):
        pose_p = estimated_seq[i]
        pose_gt_bs = gt_seq[i]
        c, R, t = umeyama_parameters(pose_p, pose_gt_bs, estimate_scale=True)
        if not np.any(np.isnan([c, R, t])):
            pose_p = pose_p * c  # Apply only scale (bone alignment)
        aligned_pose_list[i] = pose_p
    
    return aligned_pose_list


def normalize_to_standard_skeleton(poses, skeleton_model=None):
    """
    Normalize poses to a standard skeleton using bone length rescaling
    
    Args:
        poses: (N, J, 3) poses
        skeleton_model: Skeleton model for bone length normalization
        
    Returns:
        np.ndarray: Normalized poses
    """
    poses = deepcopy(np.asarray(poses))
    
    if skeleton_model is not None:
        # Use skeleton model for normalization (from baseline paper)
        for i in range(len(poses)):
            poses[i] = skeleton_model.skeleton_resize_single(
                poses[i], bone_length_file='utils/fisheye/mean3D.mat')
            print("Rescaling to Mo2Cap2")
    else:
        # Use our universal skeleton approach as fallback
        from utils.universal_skeleton import UniversalSkeletonRescaler
        rescaler = UniversalSkeletonRescaler()
        # Fit to the mean of all poses as standard skeleton
        rescaler.fit(poses)
        poses = rescaler.rescale_pose(poses, root_at_origin=True)
        
        # Ensure output is numpy array (rescale_pose might return tensor)
        if hasattr(poses, 'numpy'):
            poses = poses.numpy()
        elif not isinstance(poses, np.ndarray):
            poses = np.asarray(poses)
    
    return poses


def calculate_ba_mpjpe(estimated_seq, gt_seq, skeleton_model=None):
    """
    Calculate BA-MPJPE: Resize both sequences to standard skeleton, then PA-MPJPE
    Uses existing batch_compute_similarity_transform_torch for PA-MPJPE calculation
    
    Args:
        estimated_seq: (N, J, 3) or (B, T, J, 3) estimated poses (numpy)
        gt_seq: (N, J, 3) or (B, T, J, 3) ground truth poses (numpy)
        skeleton_model: Optional skeleton model for bone length normalization
        
    Returns:
        float: Mean BA-MPJPE in mm
    """
    import torch
    from utils.util import batch_compute_similarity_transform_torch
    from utils.loss import LossFuncMPJPE
    
    # Ensure inputs are numpy arrays
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    
    # Reshape to (N, J, 3) if needed
    if estimated_seq.ndim == 4:
        B, T, J, C = estimated_seq.shape
        estimated_seq = estimated_seq.reshape(-1, J, C)
        gt_seq = gt_seq.reshape(-1, J, C)
    
    print(f"Debug: Input shapes - estimated: {estimated_seq.shape}, gt: {gt_seq.shape}")
    
    # Step 1: Normalize BOTH sequences to the SAME standard skeleton
    if skeleton_model is not None:
        print("Debug: Using Mo2Cap2 skeletal approach")
        # Use skeleton model for normalization (from baseline paper)
        estimated_normalized = np.zeros_like(estimated_seq)
        gt_normalized = np.zeros_like(gt_seq)
        
        for i in range(len(estimated_seq)):
            estimated_normalized[i] = skeleton_model.skeleton_resize_single(
                estimated_seq[i], bone_length_file='utils/fisheye/mean3D.mat')
        for i in range(len(gt_seq)):
            gt_normalized[i] = skeleton_model.skeleton_resize_single(
                gt_seq[i], bone_length_file='utils/fisheye/mean3D.mat')
    else:
        print("Debug: Using UniversalSkeletonRescaler approach")
        # Use our universal skeleton approach as fallback
        from utils.universal_skeleton import UniversalSkeletonRescaler
        rescaler = UniversalSkeletonRescaler()
        
        # Fit to COMBINED poses to get same standard skeleton for both
        combined_poses = np.concatenate([estimated_seq, gt_seq], axis=0)
        rescaler.fit(combined_poses)
        
        # Apply same rescaling to both sequences
        estimated_tensor = rescaler.rescale_pose(estimated_seq, root_at_origin=True)
        gt_tensor = rescaler.rescale_pose(gt_seq, root_at_origin=True)
        
        # Convert tensors to numpy arrays (rescale_pose always returns tensors)
        estimated_normalized = estimated_tensor.detach().cpu().numpy()
        gt_normalized = gt_tensor.detach().cpu().numpy()
        
        print(f"Debug: After rescaling - estimated type: {type(estimated_normalized)}, shape: {estimated_normalized.shape}")
        print(f"Debug: After rescaling - gt type: {type(gt_normalized)}, shape: {gt_normalized.shape}")
    
    # Step 2: Use existing PA-MPJPE implementation
    # Ensure both are numpy arrays
    if isinstance(estimated_normalized, torch.Tensor):
        print("Debug: Converting estimated_normalized from tensor to numpy")
        estimated_normalized = estimated_normalized.detach().cpu().numpy()
    if isinstance(gt_normalized, torch.Tensor):
        print("Debug: Converting gt_normalized from tensor to numpy")
        gt_normalized = gt_normalized.detach().cpu().numpy()
    
    print(f"Debug: Final types - estimated: {type(estimated_normalized)}, gt: {type(gt_normalized)}")
    
    # Convert to torch tensors for your existing functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimated_torch = torch.from_numpy(estimated_normalized).float().to(device)
    gt_torch = torch.from_numpy(gt_normalized).float().to(device)
    
    # Use your existing PA-MPJPE implementation
    S1_hat = batch_compute_similarity_transform_torch(estimated_torch, gt_torch)
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    ba_mpjpe = mpjpe_loss_func(S1_hat, gt_torch)
    
    return ba_mpjpe.item()  # Convert back to float
