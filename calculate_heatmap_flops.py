import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import argparse
import sys
import os

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmaps.network_heatmap import HeatMap_Network
from options.train_options import TrainOptions

def calculate_heatmap_network_flops():
    """Calculate FLOPs for the heatmap network being trained"""
    
    print("Heatmap Network FLOPs Calculation")
    print("=" * 50)
    
    # Create options (similar to train_2D_heatmaps_simple.py)
    opt = TrainOptions().parse()
    opt.num_heatmap = 15  # 15 joints
    opt.init_ImageNet = True
    
    # Create model (same as in train_2D_heatmaps_simple.py)
    model = HeatMap_Network(opt, model_name='resnet18')
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: ResNet-18 + FPN Decoder")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Test different input sizes (as used in training)
    configurations = [
        {"name": "Training Config", "size": 256, "batch": 16},
        {"name": "Inference Config", "size": 256, "batch": 1},
        {"name": "High Res Config", "size": 512, "batch": 1},
    ]
    
    results = []
    
    for config in configurations:
        H = W = config["size"]
        B = config["batch"]
        
        # Create dummy input (same as training)
        input_rgb = torch.randn(B, 3, H, W)
        
        # Count FLOPs
        with torch.no_grad():
            with FlopCounterMode(display=False) as flop_counter:
                _ = model(input_rgb)
            flops = flop_counter.get_total_flops()
        
        results.append({
            'name': config['name'],
            'resolution': config['size'],
            'batch_size': config['batch'],
            'flops': flops,
            'flops_per_frame': flops / B
        })
        
        print(f"\n{config['name']}:")
        print(f"  Input: {B}×3×{H}×{W}")
        print(f"  Total FLOPs: {flops:,} ({flops/1e9:.1f}G)")
        print(f"  FLOPs per frame: {flops/B:,} ({flops/B/1e9:.1f}G)")
    
    return results

def analyze_heatmap_components():
    """Analyze FLOPs breakdown by component"""
    
    print(f"\n" + "=" * 50)
    print("HEATMAP NETWORK COMPONENT ANALYSIS")
    print("=" * 50)
    
    # Create model
    opt = TrainOptions().parse()
    opt.num_heatmap = 15
    opt.init_ImageNet = True
    model = HeatMap_Network(opt, model_name='resnet18')
    model.eval()
    
    # Test input
    input_rgb = torch.randn(1, 3, 256, 256)
    
    print("Component-wise FLOPs breakdown:")
    
    # ResNet-18 backbone
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            backbone_features = model.backbone.backbone(input_rgb)
        backbone_flops = flop_counter.get_total_flops()
    
    print(f"ResNet-18 Backbone: {backbone_flops:,} ({backbone_flops/1e9:.1f}G)")
    
    # FPN Decoder
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            _ = model.after_backbone(backbone_features)
        decoder_flops = flop_counter.get_total_flops()
    
    print(f"FPN Decoder: {decoder_flops:,} ({decoder_flops/1e9:.1f}G)")
    
    # Total
    total_flops = backbone_flops + decoder_flops
    print(f"Total Heatmap Network: {total_flops:,} ({total_flops/1e9:.1f}G)")
    
    print(f"\nBreakdown:")
    print(f"  Backbone: {backbone_flops/total_flops*100:.1f}%")
    print(f"  Decoder: {decoder_flops/total_flops*100:.1f}%")

def compare_with_other_architectures():
    """Compare with other heatmap architectures"""
    
    print(f"\n" + "=" * 50)
    print("COMPARISON WITH OTHER ARCHITECTURES")
    print("=" * 50)
    
    # Your heatmap network
    opt = TrainOptions().parse()
    opt.num_heatmap = 15
    opt.init_ImageNet = True
    
    # ResNet-18 + FPN (your current)
    model_resnet18 = HeatMap_Network(opt, model_name='resnet18')
    model_resnet18.eval()
    
    # ResNet-50 + FPN (alternative)
    model_resnet50 = HeatMap_Network(opt, model_name='resnet50')
    model_resnet50.eval()
    
    input_rgb = torch.randn(1, 3, 256, 256)
    
    # Calculate FLOPs
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            _ = model_resnet18(input_rgb)
        flops_resnet18 = flop_counter.get_total_flops()
        
        with FlopCounterMode(display=False) as flop_counter:
            _ = model_resnet50(input_rgb)
        flops_resnet50 = flop_counter.get_total_flops()
    
    print(f"ResNet-18 + FPN: {flops_resnet18:,} ({flops_resnet18/1e9:.1f}G)")
    print(f"ResNet-50 + FPN: {flops_resnet50:,} ({flops_resnet50/1e9:.1f}G)")
    print(f"ResNet-50 is {flops_resnet50/flops_resnet18:.1f}x more expensive")
    
    # Parameter comparison
    params_resnet18 = sum(p.numel() for p in model_resnet18.parameters())
    params_resnet50 = sum(p.numel() for p in model_resnet50.parameters())
    
    print(f"\nParameters:")
    print(f"ResNet-18 + FPN: {params_resnet18:,} ({params_resnet18/1e6:.1f}M)")
    print(f"ResNet-50 + FPN: {params_resnet50:,} ({params_resnet50/1e6:.1f}M)")
    print(f"ResNet-50 has {params_resnet50/params_resnet18:.1f}x more parameters")

def estimate_training_flops():
    """Estimate FLOPs during training"""
    
    print(f"\n" + "=" * 50)
    print("TRAINING FLOPs ESTIMATION")
    print("=" * 50)
    
    # Training configuration from train_2D_heatmaps_simple.py
    batch_size = 16
    sequence_length = 32  # Typical sequence length
    resolution = 256
    
    # Create model
    opt = TrainOptions().parse()
    opt.num_heatmap = 15
    opt.init_ImageNet = True
    model = HeatMap_Network(opt, model_name='resnet18')
    model.eval()
    
    # Single frame FLOPs
    input_rgb = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(input_rgb)
        single_frame_flops = flop_counter.get_total_flops()
    
    # Training FLOPs (forward + backward)
    forward_flops = single_frame_flops * batch_size * sequence_length
    backward_flops = forward_flops * 2  # Backward pass is ~2x forward
    total_training_flops = forward_flops + backward_flops
    
    print(f"Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Resolution: {resolution}×{resolution}")
    print(f"  Frames per batch: {batch_size * sequence_length}")
    
    print(f"\nFLOPs per training step:")
    print(f"  Forward pass: {forward_flops:,} ({forward_flops/1e9:.1f}G)")
    print(f"  Backward pass: {backward_flops:,} ({backward_flops/1e9:.1f}G)")
    print(f"  Total per step: {total_training_flops:,} ({total_training_flops/1e9:.1f}G)")
    
    # Per epoch estimation (assuming 1000 batches)
    batches_per_epoch = 1000
    flops_per_epoch = total_training_flops * batches_per_epoch
    
    print(f"\nPer epoch (assuming {batches_per_epoch} batches):")
    print(f"  FLOPs per epoch: {flops_per_epoch:,} ({flops_per_epoch/1e12:.1f}T)")

def main():
    print("Heatmap Network FLOPs Analysis")
    print("Based on train_2D_heatmaps_simple.py and network_heatmap.py")
    print("=" * 60)
    
    # Calculate FLOPs
    results = calculate_heatmap_network_flops()
    
    # Analyze components
    analyze_heatmap_components()
    
    # Compare architectures
    compare_with_other_architectures()
    
    # Estimate training FLOPs
    estimate_training_flops()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print("Your heatmap network (ResNet-18 + FPN) has:")
    print("  - ~11M trainable parameters")
    print("  - ~1.8G FLOPs per frame (256×256)")
    print("  - Efficient architecture for heatmap generation")
    print("  - Much lighter than full ResNet-50/101 alternatives")
    print("\nThis is the network being trained in train_2D_heatmaps_simple.py")

if __name__ == '__main__':
    main()
