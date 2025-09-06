#!/usr/bin/env python3
"""
Evaluation utilities for 3D pose estimation including universal skeleton rescaling.
"""

import torch
import numpy as np
from .universal_skeleton import create_your_skeleton_rescaler


class PoseEvaluator:
    """
    Comprehensive pose evaluator with multiple metrics including universal skeleton.
    """
    
    def __init__(self):
        self.universal_rescaler = create_your_skeleton_rescaler()
        self.is_fitted = False
        
    def fit_universal_skeleton(self, train_gt_poses):
        """
        Fit the universal skeleton from training ground truth poses.
        Call this once before evaluation.
        
        Args:
            train_gt_poses: Training GT poses of shape (..., 15, 3)
        """
        print("🔧 Fitting universal skeleton from training data...")
        self.universal_rescaler.fit(train_gt_poses)
        self.is_fitted = True
        
    def compute_mpjpe(self, pred_poses, gt_poses):
        """Standard MPJPE (Mean Per Joint Position Error)"""
        joint_errors = torch.linalg.norm(pred_poses - gt_poses, dim=-1)
        return joint_errors.mean()
        
    def compute_pa_mpjpe(self, pred_poses, gt_poses):
        """Procrustes Aligned MPJPE"""
        # Center poses
        pred_centered = pred_poses - pred_poses.mean(dim=-2, keepdim=True)
        gt_centered = gt_poses - gt_poses.mean(dim=-2, keepdim=True)
        
        # Compute optimal scale
        pred_scale = torch.linalg.norm(pred_centered.reshape(-1, 3), dim=-1).mean()
        gt_scale = torch.linalg.norm(gt_centered.reshape(-1, 3), dim=-1).mean()
        scale_factor = gt_scale / (pred_scale + 1e-8)
        
        # Apply scaling and translation
        pred_aligned = pred_centered * scale_factor + gt_poses.mean(dim=-2, keepdim=True)
        
        # Compute MPJPE
        joint_errors = torch.linalg.norm(pred_aligned - gt_poses, dim=-1)
        return joint_errors.mean()
        
    def compute_universal_mpjpe(self, pred_poses, gt_poses):
        """MPJPE with universal skeleton rescaling"""
        if not self.is_fitted:
            raise ValueError("Must call fit_universal_skeleton() first")
            
        results = self.universal_rescaler.evaluate_with_universal_skeleton(
            pred_poses, gt_poses, apply_procrustes=False
        )
        return results['mpjpe_universal']
        
    def compute_universal_pa_mpjpe(self, pred_poses, gt_poses):
        """PA-MPJPE with universal skeleton rescaling"""
        if not self.is_fitted:
            raise ValueError("Must call fit_universal_skeleton() first")
            
        results = self.universal_rescaler.evaluate_with_universal_skeleton(
            pred_poses, gt_poses, apply_procrustes=True
        )
        return results['pmpjpe_universal']
        
    def evaluate_all_metrics(self, pred_poses, gt_poses):
        """
        Compute all evaluation metrics.
        
        Args:
            pred_poses: Predicted poses (..., 15, 3)
            gt_poses: Ground truth poses (..., 15, 3)
            
        Returns:
            dict with all metrics
        """
        results = {}
        
        # Standard metrics
        results['mpjpe'] = self.compute_mpjpe(pred_poses, gt_poses).item()
        results['pa_mpjpe'] = self.compute_pa_mpjpe(pred_poses, gt_poses).item()
        
        # Universal skeleton metrics (if fitted)
        if self.is_fitted:
            results['mpjpe_universal'] = self.compute_universal_mpjpe(pred_poses, gt_poses)
            results['pa_mpjpe_universal'] = self.compute_universal_pa_mpjpe(pred_poses, gt_poses)
        
        return results


def add_universal_skeleton_to_training():
    """
    Example of how to integrate universal skeleton evaluation into your training loop.
    """
    code_example = '''
# Add this to your train.py

from utils.evaluation_utils import PoseEvaluator

# Initialize evaluator (do this once)
pose_evaluator = PoseEvaluator()

# Fit universal skeleton from training data (do this once before training)
# You need to collect some GT poses from your training data
print("Fitting universal skeleton...")
train_gt_poses = []
for i, batch in enumerate(data_loader):
    if i >= 100:  # Use first 100 batches
        break
    gt_poses = batch['gt_local_pose']  # (B, T, 15, 3)
    train_gt_poses.append(gt_poses)

train_gt_poses = torch.cat(train_gt_poses, dim=0)  # (N, T, 15, 3)
pose_evaluator.fit_universal_skeleton(train_gt_poses)

# During evaluation in your training loop
def evaluate_model(model, val_loader, pose_evaluator):
    model.eval()
    all_pred_poses = []
    all_gt_poses = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['input_rgb'].to(device)
            gt_poses = batch['gt_local_pose'].to(device)
            
            # Your model prediction
            pred_poses = model(images)  # (B, T, 15, 3)
            
            all_pred_poses.append(pred_poses.cpu())
            all_gt_poses.append(gt_poses.cpu())
    
    # Concatenate all predictions
    all_pred_poses = torch.cat(all_pred_poses, dim=0)  # (N, T, 15, 3)
    all_gt_poses = torch.cat(all_gt_poses, dim=0)      # (N, T, 15, 3)
    
    # Compute all metrics
    metrics = pose_evaluator.evaluate_all_metrics(all_pred_poses, all_gt_poses)
    
    print(f"MPJPE: {metrics['mpjpe']:.4f}")
    print(f"PA-MPJPE: {metrics['pa_mpjpe']:.4f}")
    print(f"MPJPE (Universal): {metrics['mpjpe_universal']:.4f}")
    print(f"PA-MPJPE (Universal): {metrics['pa_mpjpe_universal']:.4f}")
    
    return metrics

# In your training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Evaluate every few epochs
    if epoch % 5 == 0:
        metrics = evaluate_model(model, val_loader, pose_evaluator)
        
        # Log metrics to your loss log
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch},{metrics['mpjpe']:.4f},{metrics['pa_mpjpe']:.4f},"
                   f"{metrics['mpjpe_universal']:.4f},{metrics['pa_mpjpe_universal']:.4f}\\n")
'''
    
    print(code_example)


if __name__ == "__main__":
    print("📊 POSE EVALUATION UTILITIES")
    print("=" * 50)
    
    # Example usage
    evaluator = PoseEvaluator()
    
    # Dummy data
    batch_size, seq_len = 8, 16
    pred_poses = torch.randn(batch_size, seq_len, 15, 3)
    gt_poses = torch.randn(batch_size, seq_len, 15, 3)
    
    # Fit universal skeleton
    evaluator.fit_universal_skeleton(gt_poses)
    
    # Evaluate
    metrics = evaluator.evaluate_all_metrics(pred_poses, gt_poses)
    
    print("📊 EVALUATION RESULTS:")
    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:.4f}")
    
    print("\n💡 HOW TO INTEGRATE:")
    add_universal_skeleton_to_training()
