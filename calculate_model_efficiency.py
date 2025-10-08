import torch
import torch.nn as nn
from torchprofile import profile_macs
import argparse
import os
import sys

from action_recognition import initialize_actionformer
from utils.cross_attention_model import HeatmapToJointFeatures, PoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    """Count total parameters (including frozen ones)"""
    return sum(p.numel() for p in model.parameters())

def calculate_flops(model, input_shape):
    """
    Calculate FLOPs using torchprofile
    Args:
        model: PyTorch model
        input_shape: tuple of input dimensions (B, C, H, W) or (B, T, C, H, W)
    """
    model.eval()
    
    try:
        # Create dummy input
        if len(input_shape) == 4:  # (B, C, H, W)
            dummy_input = torch.randn(input_shape)
        elif len(input_shape) == 5:  # (B, T, C, H, W)
            dummy_input = torch.randn(input_shape)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # Calculate FLOPs
        flops = profile_macs(model, dummy_input)
        return flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None

def analyze_heatmap_network(opt):
    """Analyze heatmap network"""
    print("=== Heatmap Network Analysis ===")
    heatmap_net = HeatMap_Network(opt, model_name='resnet18')
    
    # Count parameters
    trainable_params = count_parameters(heatmap_net)
    total_params = count_total_parameters(heatmap_net)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Calculate FLOPs for single image
    input_shape = (1, 3, 256, 256)  # Single image
    flops = calculate_flops(heatmap_net, input_shape)
    if flops:
        print(f"FLOPs (single image): {flops / 1e9:.1f} G")
    
    return {
        'params': trainable_params,
        'total_params': total_params,
        'flops': flops
    }

def analyze_feature_encoder(actionformer_model):
    """Analyze feature encoder (ResNet50 + ActionFormer)"""
    print("\n=== Feature Encoder Analysis ===")
    encoder = FeatureEncoder(actionformer_model)
    
    # Count parameters
    trainable_params = count_parameters(encoder)
    total_params = count_total_parameters(encoder)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Calculate FLOPs for sequence of images
    input_shape = (1, 64, 3, 256, 256)  # (B, T, C, H, W)
    flops = calculate_flops(encoder, input_shape)
    if flops:
        print(f"FLOPs (64-frame sequence): {flops / 1e9:.1f} G")
    
    return {
        'params': trainable_params,
        'total_params': total_params,
        'flops': flops
    }

def analyze_heatmap_embedding():
    """Analyze heatmap embedding module"""
    print("\n=== Heatmap Embedding Analysis ===")
    embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=128, method='conv_pool')
    
    # Count parameters
    trainable_params = count_parameters(embedding)
    total_params = count_total_parameters(embedding)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Calculate FLOPs for heatmap sequence
    input_shape = (1, 64, 15, 64, 64)  # (B, T, J, H, W)
    flops = calculate_flops(embedding, input_shape)
    if flops:
        print(f"FLOPs (64-frame sequence): {flops / 1e9:.1f} G")
    
    return {
        'params': trainable_params,
        'total_params': total_params,
        'flops': flops
    }

# Removed spatial transformer analysis since it's not used

def analyze_pose_decoder():
    """Analyze pose decoder"""
    print("\n=== Pose Decoder Analysis ===")
    decoder = PoseDecoder(joint_dim=128, motion_dim=384, num_heads=4, num_layers=3)
    
    # Count parameters
    trainable_params = count_parameters(decoder)
    total_params = count_total_parameters(decoder)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Calculate FLOPs for joint features + motion features
    joint_features = torch.randn(1, 64, 15, 128)  # (B, T, J, D)
    motion_features = torch.randn(1, 64, 384)  # (B, T, motion_dim)
    
    # Create a wrapper to calculate FLOPs
    class DecoderWrapper(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        
        def forward(self, joint_features, motion_features):
            return self.decoder(joint_features, motion_features)
    
    wrapper = DecoderWrapper(decoder)
    flops = calculate_flops(wrapper, (joint_features, motion_features))
    if flops:
        print(f"FLOPs (64-frame sequence): {flops / 1e9:.1f} G")
    
    return {
        'params': trainable_params,
        'total_params': total_params,
        'flops': flops
    }

def analyze_full_model(opt, actionformer_model):
    """Analyze the complete model pipeline"""
    print("\n=== Full Model Pipeline Analysis ===")
    
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
    
    # Count total trainable parameters
    total_trainable = (count_parameters(encoder) + 
                      count_parameters(heatmap_embedding) + 
                      count_parameters(pose_decoder))
    
    total_params = (count_total_parameters(heatmap_net) + 
                   count_total_parameters(encoder) + 
                   count_total_parameters(heatmap_embedding) + 
                   count_total_parameters(pose_decoder))
    
    print(f"Total trainable parameters: {total_trainable:,}")
    print(f"Total parameters (including frozen): {total_params:,}")
    
    # Note: Full pipeline FLOPs would require more complex analysis
    # due to the sequential nature and different input sizes
    
    return {
        'trainable_params': total_trainable,
        'total_params': total_params
    }

def create_efficiency_table(components):
    """Create a formatted efficiency table"""
    print("\n" + "="*80)
    print("MODEL EFFICIENCY ANALYSIS")
    print("="*80)
    
    print(f"{'Component':<25} {'Params (M)':<12} {'FLOPs (G)':<12}")
    print("-" * 50)
    
    total_params = 0
    total_flops = 0
    
    for name, stats in components.items():
        if stats['params']:
            params_m = stats['params'] / 1e6
            total_params += stats['params']
            
            if stats['flops']:
                flops_g = stats['flops'] / 1e9
                total_flops += stats['flops']
                print(f"{name:<25} {params_m:<12.1f} {flops_g:<12.1f}")
            else:
                print(f"{name:<25} {params_m:<12.1f} {'N/A':<12}")
        else:
            print(f"{name:<25} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 50)
    print(f"{'TOTAL (Trainable)':<25} {total_params/1e6:<12.1f} {total_flops/1e9:<12.1f}")
    
    return total_params, total_flops

def main():
    parser = argparse.ArgumentParser(description="Calculate model efficiency metrics")
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    args = parser.parse_args()
    
    # Initialize ActionFormer
    actionformer_model = initialize_actionformer(config_file_path=args.config_path)
    
    # Initialize options (minimal for heatmap network)
    class Opt:
        def __init__(self):
            self.init_ImageNet = True
            self.num_heatmap = 15
    
    opt = Opt()
    
    # Analyze each component
    components = {}
    
    # Heatmap network (frozen during training)
    heatmap_stats = analyze_heatmap_network(opt)
    components['Heatmap Network'] = heatmap_stats
    
    # Feature encoder (ResNet50 + ActionFormer)
    encoder_stats = analyze_feature_encoder(actionformer_model)
    components['Feature Encoder'] = encoder_stats
    
    # Heatmap embedding
    embedding_stats = analyze_heatmap_embedding()
    components['Heatmap Embedding'] = embedding_stats
    
    # Pose decoder
    decoder_stats = analyze_pose_decoder()
    components['Pose Decoder'] = decoder_stats
    
    # Full model analysis
    full_stats = analyze_full_model(opt, actionformer_model)
    
    # Create efficiency table
    total_params, total_flops = create_efficiency_table(components)
    
    print(f"\n=== SUMMARY FOR PAPER ===")
    print(f"Total Trainable Parameters: {total_params/1e6:.1f} M")
    print(f"Total FLOPs: {total_flops/1e9:.1f} G")
    print(f"Total Parameters (including frozen): {full_stats['total_params']/1e6:.1f} M")
    
    # Compare with other methods (from your table)
    print(f"\n=== COMPARISON WITH OTHER METHODS ===")
    print("UnrealEgo Dataset:")
    print("  EgoGlass:     107.3M params, 16.1G FLOPs")
    print("  UnrealEgo:    106.8M params, 27.1G FLOPs") 
    print("  Ego3DPose:    178.4M params, 55.6G FLOPs")
    print(f"  Ours:         {total_params/1e6:.1f}M params, {total_flops/1e9:.1f}G FLOPs")
    
    print("\nSceneEgo Dataset:")
    print("  SceneEgo:     45.9M params, 157.3G FLOPs")
    print(f"  Ours:         {total_params/1e6:.1f}M params, {total_flops/1e9:.1f}G FLOPs")

if __name__ == '__main__':
    main()
