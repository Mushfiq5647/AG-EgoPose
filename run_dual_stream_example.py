#!/usr/bin/env python3
"""
Example usage script for Dual Stream Pose Estimation Model

This script demonstrates how to use the dual stream model with different configurations
and provides examples for training, evaluation, and inference.
"""

import os
import torch
import numpy as np
from dual_stream_config import get_config
from utils.cross_attention_model import create_dual_stream_model
from action_recognition import initialize_actionformer


def demo_model_creation():
    """Demonstrate model creation with different configurations"""
    
    print("=== DUAL STREAM MODEL CREATION DEMO ===\n")
    
    # Initialize ActionFormer (you need the actual config file)
    config_path = 'actionformer/config/ego4D_egovlp.yaml'
    
    if os.path.exists(config_path):
        actionformer_model = initialize_actionformer(config_path)
        print("✓ ActionFormer initialized successfully")
    else:
        print("⚠ ActionFormer config not found, using dummy model")
        actionformer_model = None
    
    # Test different configurations
    configs = ['default', 'lightweight', 'high_capacity']
    
    for config_name in configs:
        print(f"\n--- {config_name.upper()} Configuration ---")
        
        config = get_config(config_name)
        
        if actionformer_model is not None:
            model = create_dual_stream_model(actionformer_model, config.to_dict())
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ Model created successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Joint feature dim: {config.joint_feature_dim}")
            print(f"  Num attention heads: {config.num_heads}")
            print(f"  Transformer layers: {config.num_transformer_layers}")
        else:
            print(f"✓ Configuration loaded: {config_name}")
            print(f"  Would create model with {config.joint_feature_dim}D joint features")


def demo_forward_pass():
    """Demonstrate forward pass with dummy data"""
    
    print("\n=== FORWARD PASS DEMO ===\n")
    
    # Create dummy data
    batch_size, seq_len, num_joints = 2, 8, 15
    motion_dim, heatmap_size = 384, 128
    
    # Dummy inputs
    image_features = torch.randn(batch_size, seq_len, motion_dim)
    heatmaps = torch.randn(batch_size, seq_len, num_joints, heatmap_size, heatmap_size)
    
    print(f"Input shapes:")
    print(f"  Image features: {image_features.shape}")
    print(f"  Heatmaps: {heatmaps.shape}")
    
    # Test with lightweight config (faster for demo)
    config = get_config('lightweight')
    
    # For demo, create a minimal version without ActionFormer dependency
    try:
        from utils.cross_attention_model import (
            HeatmapToJointFeatures, 
            SpatioJointTransformer,
            MotionPoseCrossAttention,
            JointAggregator
        )
        
        print("\n--- Testing Individual Components ---")
        
        # Test heatmap converter
        heatmap_converter = HeatmapToJointFeatures(
            heatmap_size=heatmap_size,
            feature_dim=config.joint_feature_dim,
            method='conv_pool'
        )
        joint_features = heatmap_converter(heatmaps)
        print(f"✓ Heatmap conversion: {heatmaps.shape} -> {joint_features.shape}")
        
        # Test spatial transformer
        joint_transformer = SpatioJointTransformer(
            feature_dim=config.joint_feature_dim,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )
        enhanced_joints = joint_transformer(joint_features)
        print(f"✓ Spatial transformer: {joint_features.shape} -> {enhanced_joints.shape}")
        
        # Test cross attention
        cross_attention = MotionPoseCrossAttention(
            motion_dim=motion_dim,
            pose_dim=config.joint_feature_dim,
            num_heads=config.num_heads
        )
        attended_motion, attention_weights = cross_attention(image_features, enhanced_joints)
        print(f"✓ Cross attention: Motion {image_features.shape} + Joints {enhanced_joints.shape}")
        print(f"  -> Attended motion: {attended_motion.shape}")
        print(f"  -> Attention weights: {attention_weights.shape}")
        
        # Test joint aggregator
        aggregator = JointAggregator(
            joint_dim=config.joint_feature_dim,
            num_joints=num_joints,
            output_dim=config.joint_aggregation_output_dim,
            method='attention'
        )
        aggregated = aggregator(enhanced_joints)
        print(f"✓ Joint aggregation: {enhanced_joints.shape} -> {aggregated.shape}")
        
        print(f"\n✅ All components working correctly!")
        
        # Analyze attention patterns
        print(f"\n--- Attention Analysis ---")
        avg_attention = attention_weights.mean(dim=[0, 1])  # Average across batch and time
        top_3_joints = torch.topk(avg_attention, 3).indices.tolist()
        
        joint_names = config.joint_names[:num_joints]
        print(f"Top 3 attended joints:")
        for i, joint_idx in enumerate(top_3_joints):
            joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint_{joint_idx}"
            attention_score = avg_attention[joint_idx].item()
            print(f"  {i+1}. {joint_name}: {attention_score:.4f}")
        
    except ImportError as e:
        print(f"⚠ Could not test forward pass: {e}")


def demo_training_command():
    """Show example training commands"""
    
    print("\n=== TRAINING EXAMPLES ===\n")
    
    print("1. Basic training with default configuration:")
    print("python train_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_default \\")
    print("  --annotation_path ./data/train_annotation.pkl \\")
    print("  --heatmap_trained_path ./models/heatmap_model.pth \\")
    print("  --num_epochs 50 \\")
    print("  --batch_size 8 \\")
    print("  --learning_rate 1e-4")
    
    print("\n2. Lightweight model for faster training:")
    print("python train_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_lightweight \\")
    print("  --annotation_path ./data/train_annotation.pkl \\")
    print("  --joint_feature_dim 64 \\")
    print("  --num_heads 4 \\")
    print("  --num_transformer_layers 2 \\")
    print("  --batch_size 16")
    
    print("\n3. High capacity model for best performance:")
    print("python train_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_high_capacity \\")
    print("  --annotation_path ./data/train_annotation.pkl \\")
    print("  --heatmap_trained_path ./models/heatmap_model.pth \\")
    print("  --joint_feature_dim 256 \\")
    print("  --num_heads 16 \\")
    print("  --num_transformer_layers 6 \\")
    print("  --learning_rate 5e-5 \\")
    print("  --batch_size 4")
    
    print("\n4. Training without pre-trained heatmaps (uses GT):")
    print("python train_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_gt_heatmaps \\")
    print("  --annotation_path ./data/train_annotation.pkl \\")
    print("  --num_epochs 30")


def demo_evaluation_command():
    """Show example evaluation commands"""
    
    print("\n=== EVALUATION EXAMPLES ===\n")
    
    print("1. Evaluate with predicted heatmaps:")
    print("python evaluate_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_default \\")
    print("  --checkpoint_path ./models/dual_stream_default/dual_stream_epoch_50.pth \\")
    print("  --annotation_path ./data/test_annotation.pkl \\")
    print("  --heatmap_trained_path ./models/heatmap_model.pth")
    
    print("\n2. Evaluate with ground truth heatmaps:")
    print("python evaluate_dual_stream.py \\")
    print("  --model_path ./models/dual_stream_default \\")
    print("  --checkpoint_path ./models/dual_stream_default/dual_stream_epoch_50.pth \\")
    print("  --annotation_path ./data/test_annotation.pkl")


def demo_model_comparison():
    """Compare different model configurations"""
    
    print("\n=== MODEL CONFIGURATION COMPARISON ===\n")
    
    configs = {
        'Lightweight': get_config('lightweight'),
        'Default': get_config('default'),
        'High Capacity': get_config('high_capacity')
    }
    
    print(f"{'Config':<15} {'Joint Dim':<10} {'Heads':<6} {'Layers':<7} {'LR':<10} {'Est. Speed':<12}")
    print("-" * 70)
    
    for name, config in configs.items():
        est_speed = "Fast" if config.joint_feature_dim <= 64 else "Medium" if config.joint_feature_dim <= 128 else "Slow"
        print(f"{name:<15} {config.joint_feature_dim:<10} {config.num_heads:<6} "
              f"{config.num_transformer_layers:<7} {config.learning_rate:<10} {est_speed:<12}")
    
    print(f"\nRecommendations:")
    print(f"• Lightweight: For quick experiments and resource-constrained environments")
    print(f"• Default: Good balance of performance and speed for most applications")
    print(f"• High Capacity: For maximum performance when computational resources allow")


def demo_attention_interpretation():
    """Demonstrate attention interpretation"""
    
    print("\n=== ATTENTION INTERPRETATION GUIDE ===\n")
    
    print("The cross attention mechanism learns which joints are important for different motions:")
    print()
    
    examples = [
        ("Walking", ["Hip", "Knee", "Ankle"], "Leg joints get high attention for locomotion"),
        ("Reaching", ["Shoulder", "Elbow", "Wrist"], "Arm joints are crucial for manipulation"),
        ("Sitting", ["Hip", "Torso", "Knee"], "Core and leg joints for postural changes"),
        ("Jumping", ["Hip", "Knee", "Ankle", "Torso"], "Full body coordination needed")
    ]
    
    for motion, key_joints, explanation in examples:
        print(f"• {motion}:")
        print(f"  Key joints: {', '.join(key_joints)}")
        print(f"  Why: {explanation}")
        print()
    
    print("During evaluation, you can:")
    print("• Visualize attention heatmaps to see which joints the model focuses on")
    print("• Analyze attention variance to understand consistency")
    print("• Compare attention patterns between different activities")
    print("• Use attention weights to debug model behavior")


def main():
    """Run all demonstrations"""
    
    print("🚀 DUAL STREAM POSE ESTIMATION MODEL DEMO")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_model_creation()
        demo_forward_pass()
        demo_training_command()
        demo_evaluation_command()
        demo_model_comparison()
        demo_attention_interpretation()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your dataset and annotations")
        print("2. Train a 2D heatmap network (optional but recommended)")
        print("3. Train the dual stream model using train_dual_stream.py")
        print("4. Evaluate the model using evaluate_dual_stream.py")
        print("5. Analyze attention patterns for insights")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
