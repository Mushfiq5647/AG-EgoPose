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
print("Using device:", device)

def main(args):
    # Create output directory
    heatmap_dir = './utils/trained_heatmaps/mse'
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
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Model
    model = HeatMap_Network(opt, model_name='resnet18').to(device)
    
    # Simple loss and optimizer
    criterion = nn.MSELoss()
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
            
            # MEMORY OPTIMIZATION: Process heatmaps in smaller chunks
            chunk_size = 16  # Process 4 frames at a time to reduce memory usage
            all_pred_heatmaps = []
            
            for t in range(0, T, chunk_size):
                end_t = min(t + chunk_size, T)
                img_chunk = images[:, t:end_t].contiguous()  # [B, chunk_size, 3, H, W]
                hm_chunk = heatmaps[:, t:end_t].contiguous()  # [B, chunk_size, 15, H, W]
                
                # Flatten for model processing
                img_chunk_flat = img_chunk.view(-1, 3, args.crop_size, args.crop_size)
                hm_chunk_flat = hm_chunk.view(-1, 15, 128, 128)
                
                # Forward pass on chunk with mixed precision (fix deprecated warning)
                with autocast(device_type='cuda'):
                    pred_chunk = model(img_chunk_flat)
                all_pred_heatmaps.append(pred_chunk)

            
            # Concatenate all predictions
            pred_heatmaps = torch.cat(all_pred_heatmaps, dim=0)  # [B*T, 15, 128, 128]
            heatmaps_flat = heatmaps.view(B*T, 15, 128, 128)
            
            # Convert predictions to FP32 for loss computation
            pred_heatmaps = pred_heatmaps.float()
            heatmaps_flat = heatmaps_flat.float()
            
            # Loss computation (in FP32 for stability)
            loss = criterion(pred_heatmaps, heatmaps_flat)
            
            # Debug: Check for NaN/Inf and gradient norms
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected at step {i}")
                continue
                
            # Backward pass with mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Check gradient norms and NaN/Inf before stepping
            total_norm = 0
            has_nan_grad = False
            has_inf_grad = False
            
            for p in model.parameters():
                if p.grad is not None:
                    # Check for NaN/Inf in gradients
                    if torch.isnan(p.grad).any():
                        has_nan_grad = True
                    if torch.isinf(p.grad).any():
                        has_inf_grad = True
                    
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Log gradient issues and skip problematic steps
            if has_nan_grad:
                print(f"WARNING: NaN gradients detected at step {i} - SKIPPING")
                scaler.update()
                continue
            if has_inf_grad:
                print(f"WARNING: Inf gradients detected at step {i} - SKIPPING")
                scaler.update()
                continue
            
            # Skip step if gradients are too small (indicates no learning)
            if total_norm < 1e-8:
                print(f"WARNING: Very small gradients ({total_norm:.2e}) at step {i}")
                scaler.update()
                continue
            
            # Skip step if gradients are too large (indicates instability)
            if total_norm > 1e6:  # 1 million
                print(f"WARNING: Very large gradients ({total_norm:.2e}) at step {i} - SKIPPING")
                scaler.update()
                continue
            
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--clip_value', type=float, default=10.0)
    parser.add_argument('--num_epochs', type=int, default=25)  # Reasonable for heatmaps
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)  # Save every 5 epochs


    # Data
    parser.add_argument('--crop_size', type=int, default=224)
    
    args = parser.parse_args()
    main(args)
