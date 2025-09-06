#!/usr/bin/env python3
"""
Universal Skeleton Rescaling for 3D Pose Evaluation
Rescales bone lengths to a canonical "universal" skeleton for fair comparison.
"""

import torch
import numpy as np


class UniversalSkeletonRescaler:
    """
    Rescales predicted poses to match universal skeleton bone lengths.
    Used for evaluation in 3D pose estimation following common practices.
    """
    
    def __init__(self, joint_names=None, kinematic_parents=None):
        """
        Initialize the rescaler with skeleton definition.
        
        Args:
            joint_names: List of joint names (15 joints)
            kinematic_parents: List of parent indices for each joint (-1 for root)
        """
        # Your 15-joint skeleton definition
        if joint_names is None:
            self.joint_names = [
                "Neck",           # 0  - Root joint
                "Right_shoulder", # 1
                "Right_elbow",    # 2  
                "Right_wrist",    # 3
                "Left_shoulder",  # 4
                "Left_elbow",     # 5
                "Left_wrist",     # 6
                "Right_hip",      # 7
                "Right_knee",     # 8
                "Right_ankle",    # 9
                "Right_foot",     # 10
                "Left_hip",       # 11
                "Left_knee",      # 12
                "Left_ankle",     # 13
                "Left_foot"       # 14
            ]
        else:
            self.joint_names = joint_names
            
        if kinematic_parents is None:
            # Your kinematic chain with root at Neck (index 0)
            self.kinematic_parents = [-1, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
        else:
            self.kinematic_parents = kinematic_parents
            
        self.num_joints = len(self.joint_names)
        self.root_joint = self.kinematic_parents.index(-1)  # Should be 0 (Neck)
        
        # Will be computed from training data
        self.canonical_bone_lengths = None
        self.is_fitted = False
        
    def get_skeleton_connections(self):
        """Get parent-child connections for visualization"""
        connections = []
        for child, parent in enumerate(self.kinematic_parents):
            if parent != -1:
                connections.append((parent, child))
        return connections
        
    def get_bone_names(self):
        """Get bone names (from parent to child)"""
        bone_names = []
        for child, parent in enumerate(self.kinematic_parents):
            if parent != -1:
                parent_name = self.joint_names[parent]
                child_name = self.joint_names[child]
                bone_names.append(f"{parent_name}->{child_name}")
            else:
                bone_names.append("ROOT")
        return bone_names
        
    def compute_bone_lengths(self, poses_3d, eps=1e-8):
        """
        Compute bone lengths from 3D poses.
        
        Args:
            poses_3d: Tensor of shape (..., J, 3) containing 3D joint positions
            eps: Small epsilon to avoid division by zero
            
        Returns:
            bone_lengths: Tensor of shape (J,) with bone length for each joint
        """
        # Ensure tensor format
        if isinstance(poses_3d, np.ndarray):
            poses_3d = torch.from_numpy(poses_3d)
            
        device = poses_3d.device
        J = poses_3d.shape[-2]
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"
        
        bone_lengths = torch.zeros(J, device=device, dtype=poses_3d.dtype)
        bone_counts = torch.zeros(J, device=device, dtype=poses_3d.dtype)
        
        for child_idx, parent_idx in enumerate(self.kinematic_parents):
            if parent_idx == -1:  # Root joint has no bone
                continue
                
            # Compute bone vector: child - parent
            bone_vector = poses_3d[..., child_idx, :] - poses_3d[..., parent_idx, :]
            bone_length = torch.linalg.norm(bone_vector, dim=-1)  # (...,)
            
            # Average across all samples
            bone_lengths[child_idx] = bone_length.mean()
            bone_counts[child_idx] = 1
            
        return bone_lengths
        
    def fit(self, gt_poses_3d):
        """
        Fit the universal skeleton by computing canonical bone lengths from ground truth poses.
        
        Args:
            gt_poses_3d: Ground truth poses of shape (..., J, 3)
        """
        print("🔧 Computing canonical bone lengths from ground truth poses...")
        
        self.canonical_bone_lengths = self.compute_bone_lengths(gt_poses_3d)
        self.is_fitted = True
        
        # Print bone length statistics
        print("\n📏 CANONICAL BONE LENGTHS:")
        print("-" * 50)
        bone_names = self.get_bone_names()
        for i, (bone_name, length) in enumerate(zip(bone_names, self.canonical_bone_lengths)):
            if self.kinematic_parents[i] != -1:  # Skip root
                print(f"{i:2d}: {bone_name:<25} {length:.4f}")
                
        print(f"\n✅ Universal skeleton fitted with {self.num_joints} joints")
        
    def rescale_pose(self, pred_poses_3d, root_at_origin=False, eps=1e-8):
        """
        Rescale predicted poses to match canonical bone lengths.
        
        Args:
            pred_poses_3d: Predicted poses of shape (..., J, 3)
            root_at_origin: Whether to move root joint to origin
            eps: Small epsilon to avoid division by zero
            
        Returns:
            rescaled_poses: Poses with canonical bone lengths
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first to compute canonical bone lengths")
            
        # Ensure tensor format
        if isinstance(pred_poses_3d, np.ndarray):
            pred_poses_3d = torch.from_numpy(pred_poses_3d)
            
        original_shape = pred_poses_3d.shape
        device = pred_poses_3d.device
        
        # Clone to avoid modifying original
        rescaled = pred_poses_3d.clone()
        
        # Move canonical lengths to same device
        canonical_lengths = self.canonical_bone_lengths.to(device)
        
        # Optionally center root at origin
        if root_at_origin:
            root_pos = rescaled[..., self.root_joint:self.root_joint+1, :]
            rescaled = rescaled - root_pos
            
        # Build children list for efficient traversal
        children = [[] for _ in range(self.num_joints)]
        for child_idx, parent_idx in enumerate(self.kinematic_parents):
            if parent_idx != -1:
                children[parent_idx].append(child_idx)
                
        # Traverse skeleton from root and rescale each bone
        stack = [self.root_joint]
        
        while stack:
            parent_idx = stack.pop()
            
            for child_idx in children[parent_idx]:
                # Get original bone vector
                bone_vector = pred_poses_3d[..., child_idx, :] - pred_poses_3d[..., parent_idx, :]
                
                # Compute bone direction (unit vector)
                bone_length = torch.linalg.norm(bone_vector, dim=-1, keepdim=True)
                bone_direction = bone_vector / torch.clamp(bone_length, min=eps)
                
                # Set new child position with canonical bone length
                target_length = canonical_lengths[child_idx]
                rescaled[..., child_idx, :] = (rescaled[..., parent_idx, :] + 
                                              bone_direction * target_length)
                
                # Add child to stack for further processing
                stack.append(child_idx)
                
        return rescaled
        
    def evaluate_with_universal_skeleton(self, pred_poses, gt_poses, apply_procrustes=False):
        """
        Evaluate poses using universal skeleton rescaling.
        
        Args:
            pred_poses: Predicted poses (..., J, 3)
            gt_poses: Ground truth poses (..., J, 3)
            apply_procrustes: Whether to apply Procrustes alignment after rescaling
            
        Returns:
            dict with evaluation metrics
        """
        # Rescale predictions to universal skeleton
        pred_rescaled = self.rescale_pose(pred_poses, root_at_origin=False)
        
        # Compute MPJPE
        joint_errors = torch.linalg.norm(pred_rescaled - gt_poses, dim=-1)
        mpjpe = joint_errors.mean()
        
        results = {
            'mpjpe_universal': mpjpe.item(),
            'per_joint_errors': joint_errors.mean(dim=tuple(range(len(joint_errors.shape)-1))).cpu().numpy()
        }
        
        if apply_procrustes:
            # Apply Procrustes alignment after universal rescaling
            pred_aligned = self.procrustes_align(pred_rescaled, gt_poses)
            joint_errors_aligned = torch.linalg.norm(pred_aligned - gt_poses, dim=-1)
            pmpjpe = joint_errors_aligned.mean()
            results['pmpjpe_universal'] = pmpjpe.item()
            
        return results
        
    def procrustes_align(self, pred_poses, gt_poses):
        """
        Apply Procrustes alignment (translation + rotation + scaling).
        Simplified version for batch processing.
        """
        # Move to same device
        if pred_poses.device != gt_poses.device:
            pred_poses = pred_poses.to(gt_poses.device)
            
        # Center both poses
        pred_centered = pred_poses - pred_poses.mean(dim=-2, keepdim=True)
        gt_centered = gt_poses - gt_poses.mean(dim=-2, keepdim=True)
        
        # For simplicity, just apply optimal translation and uniform scaling
        # Full Procrustes would need SVD for rotation
        
        # Compute optimal scale
        pred_scale = torch.linalg.norm(pred_centered, dim=(-2, -1), keepdim=True)
        gt_scale = torch.linalg.norm(gt_centered, dim=(-2, -1), keepdim=True)
        scale_factor = gt_scale / (pred_scale + 1e-8)
        
        # Apply scaling and translation
        pred_aligned = pred_centered * scale_factor + gt_poses.mean(dim=-2, keepdim=True)
        
        return pred_aligned
        
    def save_canonical_lengths(self, save_path):
        """Save canonical bone lengths to file"""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")
            
        torch.save({
            'canonical_bone_lengths': self.canonical_bone_lengths,
            'joint_names': self.joint_names,
            'kinematic_parents': self.kinematic_parents,
            'num_joints': self.num_joints,
            'root_joint': self.root_joint
        }, save_path)
        print(f"✅ Canonical bone lengths saved to {save_path}")
        
    def load_canonical_lengths(self, load_path):
        """Load canonical bone lengths from file"""
        data = torch.load(load_path)
        
        self.canonical_bone_lengths = data['canonical_bone_lengths']
        self.joint_names = data['joint_names']
        self.kinematic_parents = data['kinematic_parents']
        self.num_joints = data['num_joints']
        self.root_joint = data['root_joint']
        self.is_fitted = True
        
        print(f"✅ Canonical bone lengths loaded from {load_path}")


def create_your_skeleton_rescaler():
    """Create rescaler with your specific skeleton definition"""
    joint_names = [
        "Neck", "Right_shoulder", "Right_elbow", "Right_wrist", 
        "Left_shoulder", "Left_elbow", "Left_wrist", "Right_hip", 
        "Right_knee", "Right_ankle", "Right_foot", "Left_hip", 
        "Left_knee", "Left_ankle", "Left_foot"
    ]
    
    kinematic_parents = [-1, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    
    return UniversalSkeletonRescaler(joint_names, kinematic_parents)


if __name__ == "__main__":
    # Example usage
    print("🔬 UNIVERSAL SKELETON RESCALER")
    print("=" * 50)
    
    # Create rescaler with your skeleton
    rescaler = create_your_skeleton_rescaler()
    
    print(f"📍 Skeleton: {rescaler.num_joints} joints")
    print(f"📍 Root joint: {rescaler.root_joint} ({rescaler.joint_names[rescaler.root_joint]})")
    
    print("\n🔗 SKELETON CONNECTIONS:")
    connections = rescaler.get_skeleton_connections()
    for parent, child in connections:
        parent_name = rescaler.joint_names[parent]
        child_name = rescaler.joint_names[child]
        print(f"({parent:2d}) {parent_name:<15} -> ({child:2d}) {child_name}")
    
    # Example with dummy data
    print("\n🧪 TESTING WITH DUMMY DATA:")
    batch_size, seq_len = 2, 4
    dummy_gt = torch.randn(batch_size, seq_len, 15, 3)
    dummy_pred = torch.randn(batch_size, seq_len, 15, 3)
    
    # Fit universal skeleton
    rescaler.fit(dummy_gt)
    
    # Evaluate
    results = rescaler.evaluate_with_universal_skeleton(dummy_pred, dummy_gt, apply_procrustes=True)
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"MPJPE (Universal): {results['mpjpe_universal']:.4f}")
    if 'pmpjpe_universal' in results:
        print(f"PA-MPJPE (Universal): {results['pmpjpe_universal']:.4f}")
    
    print("\n✅ Universal skeleton rescaler ready for use!")
