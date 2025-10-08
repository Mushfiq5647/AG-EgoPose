import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import argparse
import os

from action_recognition import initialize_actionformer
from utils.cross_attention_model import HeatmapToJointFeatures, PoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    """Count all parameters"""
    return sum(p.numel() for p in model.parameters())

def analyze_model_component(model, input_tensor, name):
    """Analyze a single model component"""
    print(f"\n=== {name} ===")
    
    # Count parameters
    trainable_params = count_parameters(model)
    total_params = count_all_parameters(model)
    
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Count FLOPs
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            if isinstance(input_tensor, tuple):
                _ = model(*input_tensor)
            else:
                _ = model(input_tensor)
        
        flops = flop_counter.get_total_flops()
        print(f"FLOPs: {flops:,} ({flops/1e9:.1f}G)")
    
    return {
        'trainable_params': trainable_params,
        'total_params': total_params,
        'flops': flops
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    args = parser.parse_args()
    
    print("Analyzing Your Model Architecture...")
    print("="*60)
    
    # Initialize ActionFormer
    actionformer_model = initialize_actionformer(config_file_path=args.config_path)
    
    # Initialize options
    class Opt:
        def __init__(self):
            self.init_ImageNet = True
            self.num_heatmap = 15
    
    opt = Opt()
    
    # Initialize all components
    heatmap_net = HeatMap_Network(opt, model_name='resnet18')
    encoder = FeatureEncoder(actionformer_model)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=128, method='conv_pool')
    pose_decoder = PoseDecoder(joint_dim=128, motion_dim=384, num_heads=4, num_layers=3)
    
    # Freeze components as in training
    for param in heatmap_net.parameters():
        param.requires_grad = False
    for param in actionformer_model.parameters():
        param.requires_grad = False
    
    # Note: FeatureEncoder has some trainable layers (layer4, linear, bn)
    # The encoder will automatically handle freezing based on its internal logic
    
    # Analyze each component
    results = {}
    
    # 1. Heatmap Network (frozen)
    heatmap_input = torch.randn(1, 3, 256, 256)
    results['Heatmap Network'] = analyze_model_component(heatmap_net, heatmap_input, "Heatmap Network (Frozen)")
    
    # 2. Feature Encoder
    encoder_input = torch.randn(1, 64, 3, 256, 256)  # (B, T, C, H, W)
    results['Feature Encoder'] = analyze_model_component(encoder, encoder_input, "Feature Encoder")
    
    # 3. Heatmap Embedding
    embedding_input = torch.randn(1, 64, 15, 64, 64)  # (B, T, J, H, W)
    results['Heatmap Embedding'] = analyze_model_component(heatmap_embedding, embedding_input, "Heatmap Embedding")
    
    # 4. Pose Decoder
    joint_features = torch.randn(1, 64, 15, 128)
    motion_features = torch.randn(1, 64, 384)
    results['Pose Decoder'] = analyze_model_component(pose_decoder, (joint_features, motion_features), "Pose Decoder")
    
    # Calculate totals
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_trainable = sum(r['trainable_params'] for r in results.values())
    total_flops = sum(r['flops'] for r in results.values())
    
    print(f"Total Trainable Parameters: {total_trainable:,} ({total_trainable/1e6:.1f}M)")
    print(f"Total FLOPs: {total_flops:,} ({total_flops/1e9:.1f}G)")
    
    # Create table format
    print(f"\n{'Component':<20} {'Params (M)':<12} {'FLOPs (G)':<12}")
    print("-" * 45)
    for name, stats in results.items():
        params_m = stats['trainable_params'] / 1e6
        flops_g = stats['flops'] / 1e9
        print(f"{name:<20} {params_m:<12.1f} {flops_g:<12.1f}")
    print("-" * 45)
    print(f"{'TOTAL':<20} {total_trainable/1e6:<12.1f} {total_flops/1e9:<12.1f}")
    
    # Compare with other methods
    print(f"\n=== COMPARISON WITH OTHER METHODS ===")
    print("UnrealEgo Dataset:")
    print("  EgoGlass:     107.3M params, 16.1G FLOPs")
    print("  UnrealEgo:    106.8M params, 27.1G FLOPs")
    print("  Ego3DPose:    178.4M params, 55.6G FLOPs")
    print(f"  Ours:         {total_trainable/1e6:.1f}M params, {total_flops/1e9:.1f}G FLOPs")
    
    print("\nSceneEgo Dataset:")
    print("  SceneEgo:     45.9M params, 157.3G FLOPs")
    print(f"  Ours:         {total_trainable/1e6:.1f}M params, {total_flops/1e9:.1f}G FLOPs")
    
    # Efficiency analysis
    print(f"\n=== EFFICIENCY ANALYSIS ===")
    print(f"Your model is {107.3/(total_trainable/1e6):.1f}x more parameter-efficient than EgoGlass")
    print(f"Your model is {27.1/(total_flops/1e9):.1f}x more FLOP-efficient than UnrealEgo")
    print(f"Your model is {55.6/(total_flops/1e9):.1f}x more FLOP-efficient than Ego3DPose")
    print(f"Your model is {157.3/(total_flops/1e9):.1f}x more FLOP-efficient than SceneEgo")

if __name__ == '__main__':
    main()
