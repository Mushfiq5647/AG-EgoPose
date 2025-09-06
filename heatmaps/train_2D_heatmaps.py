import argparse
import os
import sys
import socket
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler


# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from heatmaps.network_heatmap import HeatMap_Network

print("Running on host:", socket.gethostname())

torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
heatmap_dir = '../utils/trained_heatmaps/mse'
print("Selected device:", device)
if torch.cuda.is_available():
    print("Torch sees device as:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


class FocalLoss(nn.Module):
    """Focal Loss for better heatmap training"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Calculate focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HeatmapLoss(nn.Module):
    """Combined loss for heatmap training"""
    def __init__(self, mse_weight=0.5, focal_weight=0.3, iou_weight=0.2):
        super(HeatmapLoss, self).__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
    def iou_loss(self, pred, target):
        """IoU loss for better localization"""
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        
        # Calculate IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.focal_weight * focal + 
                     self.iou_weight * iou)
        
        return total_loss, {'mse': mse.item(), 'focal': focal.item(), 'iou': iou.item()}


def calculate_heatmap_metrics(pred, target):
    """Calculate heatmap evaluation metrics"""
    # Work with raw predictions (no sigmoid needed for MSE loss)
    
    # Peak accuracy
    pred_peaks = torch.argmax(pred.view(pred.size(0), pred.size(1), -1), dim=2)
    target_peaks = torch.argmax(target.view(target.size(0), target.size(1), -1), dim=2)
    
    # Convert to 2D coordinates
    h, w = pred.size(-2), pred.size(-1)
    pred_y = pred_peaks // w
    pred_x = pred_peaks % w
    target_y = target_peaks // w
    target_x = target_peaks % w
    
    # Calculate peak distance
    peak_distance = torch.sqrt((pred_x.float() - target_x.float())**2 + 
                              (pred_y.float() - target_y.float())**2)
    
    # Confidence scores
    pred_confidence = torch.max(pred.view(pred.size(0), pred.size(1), -1), dim=2)[0]
    
    # Safety check for confidence
    if torch.isnan(pred_confidence).any():
        mean_confidence = 0.0
    else:
        mean_confidence = pred_confidence.mean().item()
    
    # Safety check for peak distance  
    if torch.isnan(peak_distance).any():
        mean_peak_distance = 0.0
    else:
        mean_peak_distance = peak_distance.mean().item()
    
    return {
        'mean_peak_distance': mean_peak_distance,
        'mean_confidence': mean_confidence
    }


def main(args):
    # Get timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🚀 Starting training run: {timestamp}")

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    opt = TrainOptions().parse()
    train_loader = dataloader_full(opt, transform, mode='train')
    
    print("Data Loading complete")
    print(f"Training samples: {len(train_loader.dataset)}")

    J = args.num_joints
    model = HeatMap_Network(opt, model_name=args.backbone).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- loss/optim ---
    # Simple MSE Loss - stable and no NaN issues
    mse_loss = nn.MSELoss()
    
    def criterion(pred_logits, target):
        # Direct MSE loss on raw logits (no sigmoid transformation)
        loss = mse_loss(pred_logits, target)
        
        return loss, {'mse': loss.item()}
    
    # Better optimizer with different learning rates for different parts
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'backbone.backbone' in name:  # ResNet backbone layers
            backbone_params.append(param)
        else:  # after_backbone (decoder) and other parameters
            decoder_params.append(param)
    
    # Count parameters
    backbone_count = sum(p.numel() for p in backbone_params)
    decoder_count = sum(p.numel() for p in decoder_params)
    total_count = sum(p.numel() for p in model.parameters())
    
    print(f"=== PARAMETER COUNT ===")
    print(f"Backbone parameters: {backbone_count:,} ({backbone_count/total_count*100:.1f}%)")
    print(f"Decoder parameters:  {decoder_count:,} ({decoder_count/total_count*100:.1f}%)")
    print(f"Total parameters:    {total_count:,}")
    print(f"=======================")
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate * 0.1},  # Very conservative backbone LR
        {'params': decoder_params, 'lr': args.learning_rate}     # Much lower decoder LR
    ], weight_decay=1e-4)
    
    # Simple learning rate scheduler: start at 0.001, end at 0.0001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,  # Use actual epochs
            eta_min=1e-4  # End at 0.0001
        )
    
    # Loss logging
    os.makedirs('logs', exist_ok=True)
    loss_log_path = os.path.join('logs', 'loss_log_2D_heatmaps.txt')
    
    # Write header only if file doesn't exist
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, 'w') as f:
            f.write('epoch,train_loss,train_mse,mean_confidence,learning_rate\n')
    else:
        print(f"📄 Appending to existing log file: {loss_log_path}")

    total_step = len(train_loader)
    best_train_loss = float('inf')
    best_epoch = 0
    
    # Training history for plotting
    train_losses = []
    confidences = []
    
    # Mixed precision with GradScaler for stability
    scaler = GradScaler()
    


    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_metrics = {'mse': []}
        epoch_confidences = []
        
        print(f"=== Epoch {epoch+1}/{args.num_epochs} ===")
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, ncols=100)
        for i, batch in enumerate(train_pbar):
            images = batch['input_rgb'].to(device)           # (B,T,3,H,W)
            heatmaps = batch['gt_heatmap'].to(device).float()# (B,T,J,128,128)
            
            B, T, C, H, W = images.shape
            _, _, Jt, Hh, Wh = heatmaps.shape
            assert Jt == J, f"GT joints {Jt} != args.num_joints {J}"
            #
            # # Print detailed heatmap verification (first batch of first epoch only)
            # if epoch == 0 and i == 0:
            #     print("\n=== HEATMAP VERIFICATION ===")
            #     hm_min = heatmaps.min().item()
            #     hm_max = heatmaps.max().item()
            #     hm_mean = heatmaps.mean().item()
            #     hm_std = heatmaps.std().item()
            #     print(f"Overall Stats - Range: [{hm_min:.6f}, {hm_max:.6f}], Mean: {hm_mean:.6f}, Std: {hm_std:.6f}")
            #
            #     # Check if values are in expected range [0, 1]
            #     if hm_min < 0 or hm_max > 1:
            #         print(f"⚠️  WARNING: Heatmap values outside [0,1] range!")
            #
            #     # Count positive pixels (non-background)
            #     positive_pixels = (heatmaps > 0.01).sum().item()
            #     total_pixels = heatmaps.numel()
            #     pos_ratio = positive_pixels / total_pixels * 100
            #     print(f"Positive pixels (>0.01): {positive_pixels:,} / {total_pixels:,} ({pos_ratio:.2f}%)")
            #
            #     # Per-joint analysis (first 3 joints)
            #     print("Per-joint analysis:")
            #     for j in range(min(3, J)):
            #         joint_hm = heatmaps[0, 0, j]  # First batch, first time, joint j
            #         j_min = joint_hm.min().item()
            #         j_max = joint_hm.max().item()
            #         j_mean = joint_hm.mean().item()
            #         j_pos = (joint_hm > 0.01).sum().item()
            #         j_total = joint_hm.numel()
            #         print(f"  Joint {j}: Range=[{j_min:.6f}, {j_max:.6f}], Mean={j_mean:.6f}, Pos pixels={j_pos}/{j_total}")
            #
            #         # Check if this joint has any meaningful signal
            #         if j_max < 0.1:
            #             print(f"    ⚠️  Joint {j} has very low peak value ({j_max:.6f})")
            #         if j_pos == 0:
            #             print(f"    ⚠️  Joint {j} has no positive pixels!")
            #
            #     print("=== END VERIFICATION ===\n")

            # MEMORY OPTIMIZATION: Process temporal data in chunks
            chunk_size = 16  # Process 16 frames at a time
            all_preds = []
            
            for t in range(0, T, chunk_size):
                end_t = min(t + chunk_size, T)
                
                # Get chunk of images and heatmaps
                img_chunk = images[:, t:end_t]  # (B, chunk_size, C, H, W)
                
                # Flatten time for the network
                imgs_chunk_bt = img_chunk.contiguous().view(-1, C, H, W)  # (B*chunk_size, C, H, W)
                
                # Forward pass with autocast for memory efficiency
                with autocast():
                    preds_chunk = model(imgs_chunk_bt)  # (B*chunk_size, J, H, W)
                
                all_preds.append(preds_chunk)
                
                # Clear intermediate results
                del imgs_chunk_bt, preds_chunk
                torch.cuda.empty_cache()
            
            # Concatenate all predictions
            preds = torch.cat(all_preds, dim=0)  # (B*T, J, H, W)
            
            # Flatten time for ground truth
            gts_bt = heatmaps.view(B*T, J, Hh, Wh)
            
            # Loss computation in fp32 (outside autocast for stability)
            loss, loss_components = criterion(preds.float(), gts_bt.float())
            
            # Prediction verification (first batch of first epoch only)
            if epoch == 0 and i == 0:
                print("\n=== PREDICTION VERIFICATION ===")
                pred_min = preds.min().item()
                pred_max = preds.max().item()
                pred_mean = preds.mean().item()
                print(f"Raw Predictions - Range: [{pred_min:.6f}, {pred_max:.6f}], Mean: {pred_mean:.6f}")
                
                # Check predictions after sigmoid
                pred_probs = torch.sigmoid(preds)
                prob_min = pred_probs.min().item()
                prob_max = pred_probs.max().item() 
                prob_mean = pred_probs.mean().item()
                print(f"After Sigmoid - Range: [{prob_min:.6f}, {prob_max:.6f}], Mean: {prob_mean:.6f}")
                
                # Compare GT vs Pred for first joint
                gt_sample = gts_bt[0, 0]  # First batch, first joint
                pred_sample = pred_probs[0, 0]  # First batch, first joint
                print(f"Sample comparison (Joint 0):")
                print(f"  GT: min={gt_sample.min():.6f}, max={gt_sample.max():.6f}, mean={gt_sample.mean():.6f}")
                print(f"  Pred: min={pred_sample.min():.6f}, max={pred_sample.max():.6f}, mean={pred_sample.mean():.6f}")
                print("=== END PREDICTION VERIFICATION ===\n")
            
            # Calculate metrics
            metrics = calculate_heatmap_metrics(preds, gts_bt)
            
            # Better confidence metric: confidence at ground truth locations
            with torch.no_grad():
                probs = torch.sigmoid(preds)  # (B*T, J, H, W)
                gt_flat = gts_bt.view(gts_bt.size(0), gts_bt.size(1), -1)  # (B*T,J,HW)
                pred_flat = probs.view_as(gt_flat)
                gt_idx = gt_flat.argmax(-1, keepdim=True)  # (B*T,J,1)

                conf_at_gt = pred_flat.gather(-1, gt_idx).mean().item()  # 0..1
                # optional: argmax distance you’re already showing, but make sure it’s probs-based:
                pred_idx = pred_flat.argmax(-1, keepdim=True)
                H, W = probs.size(-2), probs.size(-1)
                pred_y, pred_x = (pred_idx // W).float(), (pred_idx % W).float()
                target_y, target_x = (gt_idx // W).float(), (gt_idx % W).float()
                dist_px = torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2).mean().item()

            # Backward pass with scaler
            optimizer.zero_grad()
            
            # Clear cache before backward pass
            torch.cuda.empty_cache()
            
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Normal gradient clipping for MSE stability
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Relaxed for MSE
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"WARNING: NaN/Inf gradients detected! Skipping step.")
                scaler.update()  # Update scaler even when skipping
                continue
            
            # Step optimizer with scaler
            scaler.step(optimizer)
            scaler.update()

            # Log metrics
            epoch_losses.append(loss.item())
            epoch_metrics['mse'].append(loss_components['mse'])
            epoch_confidences.append(conf_at_gt)  # Use the better confidence metric
            
            # Update progress bar every 10 steps
            if i % 10 == 0 or i == len(train_loader) - 1:
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'MSE': f"{loss_components['mse']:.4f}",
                     'Dist': f"{dist_px:.2f}",  # lower is better
                    'ConfGT': f"{conf_at_gt:.3f}"                    # higher is better
                })
        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_losses)
        avg_train_mse = np.mean(epoch_metrics['mse'])
        avg_train_conf = np.mean(epoch_confidences)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f})")
        print(f"  Train Confidence@GT: {avg_train_conf:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(heatmap_dir, 'heatmap_best.ckpt'))
            print(f"  🎉 New best model saved! (Train Loss: {best_train_loss:.6f}, Epoch: {best_epoch})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(heatmap_dir, f'heatmap-{epoch+1:03d}.ckpt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Log to file
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_train_mse:.6f},"
                   f"{avg_train_conf:.6f},{current_lr:.6f}\n")
        
        # Store for plotting  
        train_losses.append(avg_train_loss)
        confidences.append(avg_train_conf)
        
        # Step the scheduler
        scheduler.step()

    # Save final model
    final_model_path = os.path.join(heatmap_dir, 'heatmap_final.ckpt')
    torch.save(model.state_dict(), final_model_path)
    print(f"  💾 Final model saved: {final_model_path}")
    
    # Print best model info
    print(f"\n🏆 Best model was at epoch {best_epoch} with training loss: {best_train_loss:.6f}")
    print(f"Best model saved as: {os.path.join(heatmap_dir, 'heatmap_best.ckpt')}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    if best_epoch > 0:
        plt.axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(confidences, label='Mean Confidence')
    plt.title('Mean Confidence')
    plt.xlabel('Epoch')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot([optimizer.param_groups[0]['lr'] for _ in range(len(train_losses))], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save training curves in logs folder
    curves_path = os.path.join('logs', 'training_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to: {curves_path}")
    plt.show()

    # Save epoch-wise losses in a readable format with timestamp
    epoch_summary_path = os.path.join('logs', f'epoch_losses_{timestamp}.txt')
    with open(epoch_summary_path, 'w') as f:
        f.write(f"Epoch-wise Training Summary - {timestamp}\n")
        f.write("="*60 + "\n")
        f.write(f"Training Args: {args}\n")
        f.write("="*60 + "\n\n")
        f.write("Epoch | Train Loss | Confidence | LR\n")
        f.write("-"*50 + "\n")
        
        for i, (train_loss, conf) in enumerate(zip(train_losses, confidences)):
            epoch_num = i + 1
            lr_at_epoch = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 'N/A'
            
            f.write(f"{epoch_num:5d} | {train_loss:10.6f} | {conf:10.4f} | {lr_at_epoch:.6f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Train Loss: {best_train_loss:.6f}\n")
        f.write(f"Final Confidence: {confidences[-1]:.4f}\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📝 Epoch summary saved to: {epoch_summary_path}")

    print("Training complete!")
    print(f"Best training loss: {best_train_loss:.6f} at epoch {best_epoch}")
    print(f"Final mean confidence: {confidences[-1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','resnet101'])
    parser.add_argument('--num_joints', type=int, default=15)
    parser.add_argument('--init_imagenet', action='store_true', help='ResNet ImageNet init for backbone')

    # Train
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=25)  # Reasonable for heatmaps
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)  # Save every 5 epochs


    # Data
    parser.add_argument('--crop_size', type=int, default=224)

    args = parser.parse_args()
    print(args)
    main(args)
