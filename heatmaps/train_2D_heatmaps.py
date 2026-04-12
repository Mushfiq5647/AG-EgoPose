import argparse
import math
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


@torch.no_grad()
def heatmap_argmax_xy(hm):
    """Return integer argmax (x, y) per (B, J) for heatmaps shaped (B, J, H, W)."""
    B, J, H, W = hm.shape
    flat = hm.reshape(B, J, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx %  W).float()
    return xs, ys


@torch.no_grad()
def run_validation(model, val_loader, criterion, device, crop_size, num_heatmap, max_batches=None):
    """Compute average BCE loss and mean argmax pixel distance on validation set."""
    model.eval()
    total_loss = 0.0
    total_dist = 0.0
    n_batches = 0
    n_joints_seen = 0

    for vi, batch in enumerate(val_loader):
        if max_batches is not None and vi >= max_batches:
            break
        try:
            imgs_left, imgs_right, hms_left, hms_right = unpack_heatmap_batch(batch)
        except KeyError:
            continue

        imgs_left  = imgs_left.to(device)
        imgs_right = imgs_right.to(device)
        hms_left   = hms_left.to(device).float()
        hms_right  = hms_right.to(device).float()

        B, T = imgs_left.shape[:2]
        left_flat      = imgs_left.reshape(-1, 3, crop_size, crop_size)
        right_flat     = imgs_right.reshape(-1, 3, crop_size, crop_size)
        hm_left_flat   = hms_left.reshape(B * T, num_heatmap, 64, 64)
        hm_right_flat  = hms_right.reshape(B * T, num_heatmap, 64, 64)

        with autocast(device_type='cuda'):
            pred = model(left_flat, right_flat)   # (B*T, num_heatmap*2, H, W)
        pred = pred.float()
        pred_left  = pred[:, :num_heatmap]
        pred_right = pred[:, num_heatmap:]

        loss = (criterion(pred_left, hm_left_flat) + criterion(pred_right, hm_right_flat)) / 2
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1

        # Argmax pixel distance over both views
        for p_hm, gt_hm in [(pred_left, hm_left_flat), (pred_right, hm_right_flat)]:
            pred_x, pred_y = heatmap_argmax_xy(torch.sigmoid(p_hm))
            gt_x,   gt_y   = heatmap_argmax_xy(gt_hm)
            dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            total_dist    += dist.sum().item()
            n_joints_seen += dist.numel()

    model.train()
    avg_loss = total_loss / max(1, n_batches)
    avg_dist = total_dist / max(1, n_joints_seen)
    return avg_loss, avg_dist


def unpack_heatmap_batch(batch):
    """Return (imgs_left, imgs_right, hms_left, hms_right) each (B, T, *, H, W).

    For mono datasets the single view is duplicated so the stereo model still works —
    both outputs will be supervised with the same heatmap target.
    """
    if 'input_rgb_left' in batch and 'input_rgb_right' in batch:
        return (batch['input_rgb_left'],  batch['input_rgb_right'],
                batch['gt_heatmap_left'], batch['gt_heatmap_right'])

    if 'input_rgb' in batch and 'gt_heatmap' in batch:
        img = batch['input_rgb']
        hm  = batch['gt_heatmap']
        return img, img, hm, hm   # duplicate view for mono datasets

    raise KeyError(
        "Unsupported batch format. Expected stereo keys "
        "('input_rgb_left/right', 'gt_heatmap_left/right') or mono keys "
        "('input_rgb', 'gt_heatmap')."
    )


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Data loading (parses options too — gives us opt.model)
    opt = TrainOptions().parse()

    # Create output directory (depends on opt.model, so do this AFTER parse)
    if opt.model == 'unrealego':
        heatmap_dir = './utils/trained_heatmaps/bce_unrealego'
    else:
        heatmap_dir = './utils/trained_heatmaps/bce_combined'
    os.makedirs(heatmap_dir, exist_ok=True)

    # Create log file for epoch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(heatmap_dir, f'loss_log_heatmaps.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_PxDist,Best_PxDist,Learning_Rate\n")

    train_loader = dataloader_full(opt, transform, mode='train')
    try:
        val_loader = dataloader_full(opt, transform, mode='validation')
    except FileNotFoundError as e:
        print(f"⚠️  No validation list found ({e}); validation will be skipped.")
        val_loader = None
    
    # Model (now truly wired to CLI --backbone)
    model = HeatMap_Network(opt, model_name=args.backbone).to(device)
    # Explicitly ensure all params are trainable for heatmap backbone retraining
    for p in model.parameters():
        p.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Heatmap backbone: {args.backbone}")
    print(f"Heatmap params (total/trainable): {total_params:,}/{trainable_params:,}")
    
    # Loss: heatmaps are very sparse (~25 positive pixels out of 4096 per joint),
    # so use a high pos_weight to avoid soft, low-amplitude peaks.
    pos_weight = torch.tensor(args.pos_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    # Optimizer: AdamW with discriminative LR — pretrained encoder gets 0.1x LR
    # and standard ConvNeXt-style weight decay; fresh decoder gets full LR + light WD.
    encoder_params = list(model.backbone.parameters())
    decoder_params = list(model.after_backbone.parameters())
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.learning_rate * 0.1, 'weight_decay': 0.05},
        {'params': decoder_params, 'lr': args.learning_rate,        'weight_decay': 1e-4},
    ])

    # Scheduler: 1-epoch linear warmup then cosine decay to 1% of base LR.
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.num_epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler (fix deprecated warning)
    scaler = GradScaler(device='cuda')
    
    # Training: best model selected by validation localization (argmax px distance);
    # falls back to train BCE loss if no validation set is available.
    best_loss = float('inf')
    best_val_dist = float('inf')
    train_losses = []
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for i, batch in enumerate(pbar):
            imgs_left, imgs_right, hms_left, hms_right = unpack_heatmap_batch(batch)
            imgs_left  = imgs_left.to(device)   # (B, T, 3, H, W)
            imgs_right = imgs_right.to(device)
            hms_left   = hms_left.to(device).float()    # (B, T, J, 64, 64)
            hms_right  = hms_right.to(device).float()

            B, T = imgs_left.shape[:2]
            left_flat     = imgs_left.reshape(-1, 3, args.crop_size, args.crop_size)   # (B*T, 3, H, W)
            right_flat    = imgs_right.reshape(-1, 3, args.crop_size, args.crop_size)
            hm_left_flat  = hms_left.reshape(B * T, args.num_heatmap, 64, 64)
            hm_right_flat = hms_right.reshape(B * T, args.num_heatmap, 64, 64)

            # Debug: Check ground truth heatmaps (first batch only)
            if i == 0:
                print(f"GT heatmaps shape: {hm_left_flat.shape}")
                print(f"GT left  min/max/mean: {hm_left_flat.min():.4f}/{hm_left_flat.max():.4f}/{hm_left_flat.mean():.6f}")
                print(f"GT right min/max/mean: {hm_right_flat.min():.4f}/{hm_right_flat.max():.4f}/{hm_right_flat.mean():.6f}")
                print(f"GT non-zero count (left): {(hm_left_flat > 0).sum().item()}")

            # Chunked forward pass — each chunk processes a slice of B*T frames from both views.
            # Loss = average of left and right BCE, normalised by n_chunks.
            n_total = left_flat.shape[0]
            n_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
            running_loss = 0.0
            debug_pred_left = None

            optimizer.zero_grad()
            for c in range(n_chunks):
                s = c * args.chunk_size
                e = min((c + 1) * args.chunk_size, n_total)

                left_chunk     = left_flat[s:e]
                right_chunk    = right_flat[s:e]
                hm_left_chunk  = hm_left_flat[s:e]
                hm_right_chunk = hm_right_flat[s:e]

                with autocast(device_type='cuda'):
                    # Model outputs (chunk, num_heatmap*2, H, W): left then right
                    pred_chunk = model(left_chunk, right_chunk)

                pred_chunk = pred_chunk.float()
                pred_left_chunk  = pred_chunk[:, :args.num_heatmap]
                pred_right_chunk = pred_chunk[:, args.num_heatmap:]

                if i == 0 and c == 0:
                    debug_pred_left = pred_left_chunk.detach()

                chunk_loss = (
                    criterion(pred_left_chunk,  hm_left_chunk) +
                    criterion(pred_right_chunk, hm_right_chunk)
                ) / (2 * n_chunks)
                running_loss += chunk_loss.item()

                if torch.isnan(chunk_loss) or torch.isinf(chunk_loss):
                    print(f"WARNING: NaN/Inf chunk loss at step {i}, chunk {c}")
                    running_loss = None
                    break

                scaler.scale(chunk_loss).backward()

            if running_loss is None:
                scaler.update()
                continue

            # Debug predictions (first batch only)
            if debug_pred_left is not None and i == 0:
                print(f"Pred (left) min/max/mean/std: "
                      f"{debug_pred_left.min():.3f}/{debug_pred_left.max():.3f}/"
                      f"{debug_pred_left.mean():.4f}/{debug_pred_left.std():.4f}")

            loss = torch.tensor(running_loss, device=device)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at step {i}")
                continue

            scaler.unscale_(optimizer)

            # NaN/Inf gradient check
            has_nan_inf = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if has_nan_inf:
                print(f"WARNING: NaN/Inf gradients at step {i} - SKIPPING")
                scaler.update()
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_value)

            if not torch.isfinite(total_norm):
                print(f"WARNING: non-finite grad norm at step {i} - SKIPPING")
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'GradNorm': f'{total_norm:.2e}',
                'PredL': f'{debug_pred_left.min():.2f}/{debug_pred_left.max():.2f}' if debug_pred_left is not None else 'n/a',
            })
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_losses.append(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Validation: BCE + argmax pixel distance (the latter is the metric we track)
        val_loss = float('nan')
        val_dist = float('nan')
        if val_loader is not None:
            val_loss, val_dist = run_validation(
                model, val_loader, criterion, device,
                args.crop_size, args.num_heatmap, max_batches=args.val_max_batches
            )

        print(
            f'Epoch {epoch+1}: TrainLoss={avg_loss:.4f}  '
            f'ValLoss={val_loss:.4f}  ValPxDist={val_dist:.3f}  '
            f'LR={current_lr:.2e}'
        )

        # Log epoch results to file
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{val_dist:.6f},"
                f"{best_val_dist:.6f},{current_lr:.6f}\n"
            )

        # Save best model — prefer validation pixel distance, fall back to train loss
        improved = False
        if val_loader is not None and not math.isnan(val_dist):
            if val_dist < best_val_dist:
                best_val_dist = val_dist
                improved = True
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                improved = True

        if improved:
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
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(heatmap_dir, 'training_curve.png'))
    plt.close()
    
    if val_loader is not None:
        print(f'Training completed! Best val pixel distance: {best_val_dist:.4f}')
    else:
        print(f'Training completed! Best train loss: {best_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backbone',
        type=str,
        default='convnext_tiny',
        choices=['convnext_tiny', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
        help='2D heatmap backbone to train'
    )
    parser.add_argument('--num_joints', type=int, default=15)
    parser.add_argument('--num_heatmap', type=int, default=15, help='Number of heatmap channels')

    # Train
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='base LR for the decoder; encoder is trained at 0.1x this')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='linear warmup epochs before cosine decay kicks in')
    parser.add_argument('--pos_weight', type=float, default=50.0,
                        help='BCEWithLogits positive class weight (heatmaps are very sparse)')
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--chunk_size', type=int, default=64,
                        help='number of flattened frames per forward chunk to reduce VRAM')
    parser.add_argument('--val_max_batches', type=int, default=200,
                        help='cap on validation batches per epoch (None for full pass)')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)  # Save every 5 epochs


    # Data
    parser.add_argument('--crop_size', type=int, default=256)
    
    args, _ = parser.parse_known_args()
    main(args)
