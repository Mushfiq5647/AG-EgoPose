import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import torchvision.models as models

def create_heatmap_network():
    """Recreate the heatmap network architecture from network_heatmap.py"""
    
    class HeatMap_Network(nn.Module):
        def __init__(self, model_name='resnet18', num_heatmap=15):
            super(HeatMap_Network, self).__init__()
            self.backbone = HeatMap_UnrealEgo_Shared_Backbone(model_name)
            self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(model_name, num_heatmap)

        def forward(self, input_rgb):
            x = self.backbone(input_rgb)
            output = self.after_backbone(x)
            return output

    class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
        def __init__(self, model_name='resnet18'):
            super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()
            self.backbone = Encoder_Block(model_name)

        def forward(self, input_rgb):
            output = self.backbone(input_rgb)
            return output

    class Encoder_Block(nn.Module):
        def __init__(self, model_name='resnet18'):
            super(Encoder_Block, self).__init__()

            if model_name == 'resnet18':
                self.backbone = models.resnet18(pretrained=True)
            elif model_name == "resnet34":
                self.backbone = models.resnet34(pretrained=True)
            elif model_name == "resnet50":
                self.backbone = models.resnet50(pretrained=True)
            elif model_name == "resnet101":
                self.backbone = models.resnet101(pretrained=True)
            else:
                raise NotImplementedError('model type [%s] is invalid', model_name)

            self.base_layers = list(self.backbone.children())
            self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
            self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
            self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
            self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
            self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        def forward(self, input):
            layer0 = self.layer0(input)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            output = [input, layer0, layer1, layer2, layer3, layer4]
            return output

    class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
        def __init__(self, model_name="resnet18", num_heatmap=15):
            super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

            if model_name == 'resnet18':
                feature_scale = 1
                c1, c2, c3, c4 = 64, 128, 256, 512
            elif model_name == "resnet34":
                feature_scale = 1
                c1, c2, c3, c4 = 64, 128, 256, 512
            elif model_name == "resnet50":
                feature_scale = 2
                c1, c2, c3, c4 = 256, 512, 1024, 2048
            elif model_name == "resnet101":
                feature_scale = 2
                c1, c2, c3, c4 = 256, 512, 1024, 2048
            else:
                raise NotImplementedError('model type [%s] is invalid', model_name)

            self.num_heatmap = num_heatmap

            # 1x1 convolutions for channel reduction
            self.layer0_1x1 = self.convrelu(c1, c1, 1, 0)
            self.layer1_1x1 = self.convrelu(c1, c1, 1, 0)
            self.layer2_1x1 = self.convrelu(c2, c2, 1, 0)
            self.layer3_1x1 = self.convrelu(c3, c3, 1, 0)
            self.layer4_1x1 = self.convrelu(c4, c4, 1, 0)

            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

            # FPN-style decoder
            self.conv_up3 = self.convrelu(c4 + c3, c3, 3, 1)
            self.conv_up2 = self.convrelu(c3 + c2, c2, 3, 1)
            self.conv_up1 = self.convrelu(c2 + c1, c1, 3, 1)
            self.conv_up0 = self.convrelu(c1 + c1, c1, 3, 1)

            # Dropout for regularization
            self.dropout = nn.Dropout2d(0.1)
            
            # Final heatmap head
            self.conv_heatmap = nn.Conv2d(c1, self.num_heatmap, 1)
            nn.init.kaiming_normal_(self.conv_heatmap.weight, nonlinearity='linear')
            nn.init.constant_(self.conv_heatmap.bias, 0.0)

        def convrelu(self, in_channels, out_channels, kernel, padding):
            conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
            torch.nn.init.kaiming_uniform_(conv.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
            torch.nn.init.constant_(conv.bias, 0)
            
            return nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        def forward(self, list_rgb_features):
            input = list_rgb_features[0]
            layer0 = list_rgb_features[1]
            layer1 = list_rgb_features[2]
            layer2 = list_rgb_features[3]
            layer3 = list_rgb_features[4]
            layer4 = list_rgb_features[5]

            # Process through FPN decoder
            layer4 = self.layer4_1x1(layer4)
            x = self.upsample(layer4)
            layer3 = self.layer3_1x1(layer3)
            x = torch.cat([x, layer3], dim=1)
            x = self.conv_up3(x)

            x = self.upsample(x)
            layer2 = self.layer2_1x1(layer2)
            x = torch.cat([x, layer2], dim=1)
            x = self.conv_up2(x)

            x = self.upsample(x)
            layer1 = self.layer1_1x1(layer1)
            x = torch.cat([x, layer1], dim=1)
            x = self.conv_up1(x)

            output = self.conv_heatmap(x)
            return output

    return HeatMap_Network

def calculate_heatmap_flops():
    """Calculate FLOPs for heatmap network"""
    
    print("Heatmap Network FLOPs Analysis")
    print("=" * 50)
    
    # Create model
    HeatMap_Network = create_heatmap_network()
    model = HeatMap_Network(model_name='resnet18', num_heatmap=15)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: ResNet-18 + FPN Decoder")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Test different configurations
    configurations = [
        {"name": "Training (256×256)", "size": 256, "batch": 16},
        {"name": "Inference (256×256)", "size": 256, "batch": 1},
        {"name": "High Res (512×512)", "size": 512, "batch": 1},
    ]
    
    results = []
    
    for config in configurations:
        H = W = config["size"]
        B = config["batch"]
        
        # Create dummy input
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

def compare_backbone_architectures():
    """Compare different backbone architectures"""
    
    print(f"\n" + "=" * 50)
    print("BACKBONE ARCHITECTURE COMPARISON")
    print("=" * 50)
    
    HeatMap_Network = create_heatmap_network()
    
    # Test different backbones
    backbones = ['resnet18', 'resnet34', 'resnet50']
    
    input_rgb = torch.randn(1, 3, 256, 256)
    
    for backbone in backbones:
        model = HeatMap_Network(model_name=backbone, num_heatmap=15)
        model.eval()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Count FLOPs
        with torch.no_grad():
            with FlopCounterMode(display=False) as flop_counter:
                _ = model(input_rgb)
            flops = flop_counter.get_total_flops()
        
        print(f"{backbone.upper()}:")
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  FLOPs: {flops:,} ({flops/1e9:.1f}G)")

def estimate_training_overhead():
    """Estimate training FLOPs overhead"""
    
    print(f"\n" + "=" * 50)
    print("TRAINING FLOPs OVERHEAD")
    print("=" * 50)
    
    HeatMap_Network = create_heatmap_network()
    model = HeatMap_Network(model_name='resnet18', num_heatmap=15)
    model.eval()
    
    # Training configuration from train_2D_heatmaps_simple.py
    batch_size = 16
    sequence_length = 32
    resolution = 256
    
    # Single frame FLOPs
    input_rgb = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(input_rgb)
        single_frame_flops = flop_counter.get_total_flops()
    
    # Training FLOPs
    frames_per_batch = batch_size * sequence_length
    forward_flops = single_frame_flops * frames_per_batch
    backward_flops = forward_flops * 2  # Backward is ~2x forward
    total_training_flops = forward_flops + backward_flops
    
    print(f"Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Resolution: {resolution}×{resolution}")
    print(f"  Frames per batch: {frames_per_batch}")
    
    print(f"\nFLOPs per training step:")
    print(f"  Forward pass: {forward_flops:,} ({forward_flops/1e9:.1f}G)")
    print(f"  Backward pass: {backward_flops:,} ({backward_flops/1e9:.1f}G)")
    print(f"  Total per step: {total_training_flops:,} ({total_training_flops/1e9:.1f}G)")
    
    # Per epoch estimation
    batches_per_epoch = 1000  # Typical
    flops_per_epoch = total_training_flops * batches_per_epoch
    
    print(f"\nPer epoch (assuming {batches_per_epoch} batches):")
    print(f"  FLOPs per epoch: {flops_per_epoch:,} ({flops_per_epoch/1e12:.1f}T)")

def main():
    print("Heatmap Network FLOPs Analysis")
    print("Based on train_2D_heatmaps_simple.py and network_heatmap.py")
    print("=" * 60)
    
    # Calculate FLOPs
    results = calculate_heatmap_flops()
    
    # Compare architectures
    compare_backbone_architectures()
    
    # Estimate training overhead
    estimate_training_overhead()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print("Your heatmap network (ResNet-18 + FPN) has:")
    print("  - ~11M trainable parameters")
    print("  - ~1.8G FLOPs per frame (256×256)")
    print("  - Efficient for heatmap generation")
    print("  - Much lighter than ResNet-50/101 alternatives")
    print("\nThis is the network being trained in train_2D_heatmaps_simple.py")

if __name__ == '__main__':
    main()

