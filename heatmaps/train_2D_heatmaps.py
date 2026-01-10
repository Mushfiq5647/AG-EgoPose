import argparse
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from torch.amp import autocast, GradScaler

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from heatmaps.network_heatmap import HeatMap_Network

torch.manual_seed(7)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create output directory
    heatmap_dir = './utils/trained_heatmaps/bce_combined'
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Create log file for epoch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(heatmap_dir, f'loss_log_heatmaps.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Avg_Loss,Best_Loss,Learning_Rate\n")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Data loading
    opt = TrainOptions().parse()
    train_loader = dataloader_full(opt, transform, mode='train')
    
    # Model
    model = HeatMap_Network(opt, model_name='resnet18').to(device)
    
    # Simple loss and optimizer
    pos_weight = torch.tensor(12.0, device=device)
    criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    # criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Mixed precision scaler (fix deprecated warning)
    scaler = GradScaler(device='cuda')
    
    # Training
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for i, batch in enumerate(pbar):
            images = batch['input_rgb'].to(device)  # [B, T, 3, H, W]
            heatmaps = batch['gt_heatmap'].to(device)  # [B, T, 15, H, W]
            
            B, T = images.shape[:2]
            
            # Process ALL images at once (more efficient and stable)
            img_flat = images.view(-1, 3, args.crop_size, args.crop_size)  # [B*T, 3, H, W]
            heatmaps_flat = heatmaps.view(B*T, 15, 64,64)  # [B*T, 15, H, W]
            
            # Debug: Check ground truth heatmaps
            if i == 0:  # Only print for first batch
                print(f"GT heatmaps dtype: {heatmaps_flat.dtype}")
                print(f"GT heatmaps shape: {heatmaps_flat.shape}")
                print(f"GT heatmaps min/max: {heatmaps_flat.min().item():.6f}/{heatmaps_flat.max().item():.6f}")
                print(f"GT heatmaps mean: {heatmaps_flat.mean().item():.6f}")
                print(f"GT heatmaps non-zero count: {(heatmaps_flat > 0).sum().item()}")
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                pred_heatmaps = model(img_flat)  # [B*T, 15, 128, 128] - raw logits now
            
            # Debug: Check model predictions
            if i == 0:  # Only print for first batch
                print(f"Pred heatmaps dtype: {pred_heatmaps.dtype}")
                print(f"Pred heatmaps min/max: {pred_heatmaps.min().item():.6f}/{pred_heatmaps.max().item():.6f}")
                print(f"Pred heatmaps mean: {pred_heatmaps.mean().item():.6f}")
                print(f"Pred heatmaps std: {pred_heatmaps.std().item():.6f}")
            
            # Convert predictions to FP32 for loss computation
            pred_heatmaps = pred_heatmaps.float()
            heatmaps_flat = heatmaps_flat.float()
            loss = criterion(pred_heatmaps, heatmaps_flat)
            
            # Debug: Check for NaN/Inf and gradient norms
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected at step {i}")
                continue
                
            # Backward pass with mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale gradients first (required for AMP)
            scaler.unscale_(optimizer)
            
            # Check for NaN/Inf gradients AFTER unscaling
            has_nan_inf = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_inf = True
                    break
            
            if has_nan_inf:
                print(f"WARNING: NaN/Inf gradients detected at step {i} - SKIPPING")
                scaler.update()  # Update scaler even when skipping
                continue
            
            # Gradient clipping to prevent NaN/Inf
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_value)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'GradNorm': f'{total_norm:.2e}',
                'PredRange': f'{pred_heatmaps.min().item():.3f}-{pred_heatmaps.max().item():.3f}',
                'GT_Range': f'{heatmaps_flat.min().item():.3f}-{heatmaps_flat.max().item():.3f}'
            })
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}')
        
        # Log epoch results to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{best_loss:.6f},{current_lr:.6f}\n")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(heatmap_dir, 'heatmap_best.ckpt'))
            print(f'  New best model saved!')
        
        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(heatmap_dir, f'heatmap_epoch_{epoch+1}.ckpt'))
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(heatmap_dir, 'heatmap_final.ckpt'))
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(heatmap_dir, 'training_curve.png'))
    plt.close()
    
    print(f'Training completed! Best loss: {best_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','resnet101'])
    parser.add_argument('--num_joints', type=int, default=15)
    parser.add_argument('--num_heatmap', type=int, default=15, help='Number of heatmap channels')

    # Train
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # Much smaller LR to prevent gradient explosion
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--num_epochs', type=int, default=30)  # Reasonable for heatmaps
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)  # Save every 5 epochs


    # Data
    parser.add_argument('--crop_size', type=int, default=256)
    
    args = parser.parse_args()
    main(args)
