# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.amp import autocast, GradScaler

from utils.action_recognition import initialize_actionformer
from options.train_options import TrainOptions
from utils.data_loader import dataloader_full
from utils.cross_attention_model import HeatmapToJointFeatures
from utils.cross_attention_model import SpatialJointTransformer
from utils.cross_attention_model import PoseDecoder, ConcatFusionDecoder
from utils.model import MLPPoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.loss import LossFuncLimb, LossFuncMPJPE, LossFuncCosSim
from utils.model import FeatureEncoder
import socket

print("Running on host:", socket.gethostname())
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)
if torch.cuda.is_available():
    print("Current CUDA device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

num_joints = 15


def initialize_gelu_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def initialize_relu_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight,
            a=0,
            nonlinearity='relu'
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_grad_norm(model, model_name=""):
    total_norm = 0
    count = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            if param_norm > 0:
                count += 1
            total_norm += param_norm ** 2
    result = total_norm ** (1. / 2)
    if model_name and count == 0:
        print(
            f"  WARNING: {model_name} has no parameters with non-zero gradients "
            f"({sum(1 for p in model.parameters() if p.requires_grad)} total trainable params)"
        )
    return result


CKPT_BASE = '/data/My_Backup/ag-egopose-ckpt'


def main(args):
    if args.skip_temporal:
        actionformer_feature_extractor = None
        print("ABLATION: Skipping entire temporal stream (DINOv2 + ActionFormer)")
    else:
        actionformer_feature_extractor = initialize_actionformer(
            config_file_path=args.config_path,
            random_init=args.random_actionformer
        )
    # Resolve model_dir early so all path operations use the correct absolute path
    model_dir = args.model_path if os.path.isabs(args.model_path) else os.path.join(CKPT_BASE, args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    model_name = os.path.basename(model_dir)
    loss_log_path = f'loss_log_{model_name}.txt'
    with open(loss_log_path, 'w') as f:
        f.write("Epoch, Train_MPJPE, Train_Cos, Train_Bone, Train_Total, Best_Total \n")
    print(f"Loss log file: {loss_log_path}")

    best_loss = float('inf')
    best_epoch = 0

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    opt = TrainOptions().parse()
    data_loader = dataloader_full(opt, transform, mode='train')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))

    net_heatmap = HeatMap_Network(opt, model_name=args.heatmap_backbone).to(device)

    if os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        for param in net_heatmap.parameters():
            param.requires_grad = False
        net_heatmap.eval()
        print("2D heatmap network loaded and frozen")
        print(f"Using heatmap model: {args.heatmap_trained_path}")
    else:
        print(f"Warning: Pre-trained heatmap model not found at {args.heatmap_trained_path}")
        print("Will use ground truth heatmaps for training")

    heatmap_embedding = HeatmapToJointFeatures(
        heatmap_size=64,
        feature_dim=args.hm_embed_dim,
        method='conv_pool'
    ).to(device)
    if args.skip_temporal:
        encoder = None
    else:
        encoder = FeatureEncoder(actionformer_feature_extractor, skip_temporal=False).to(device)
    spatial_joint_transformer = SpatialJointTransformer(
        args.hm_embed_dim,
        num_heads=4,
        num_layers=3
    ).to(device)
    if args.mlp_decoder:
        pose_decoder = MLPPoseDecoder(motion_dim=384, joint_dim=args.hm_embed_dim).to(device)
        print("ABLATION: Using MLPPoseDecoder (concat + MLP)")
    elif args.concat_fusion:
        pose_decoder = ConcatFusionDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
        print("ABLATION: Using ConcatFusionDecoder (concat + self-attention)")
    else:
        pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3,
                                   use_mcjq=not args.skip_mcjq).to(device)
        if args.skip_mcjq:
            print("ABLATION: Skipping MCJQ — using static learned queries only")
    print("Models initialized")

    af_unfrozen_params = []
    if not args.skip_temporal and not args.freeze_actionformer:
        backbone = actionformer_feature_extractor.model.backbone

        for p in actionformer_feature_extractor.input_proj.parameters():
            p.requires_grad = True
            af_unfrozen_params.append(p)

        for p in backbone.embd.parameters():
            p.requires_grad = True
            af_unfrozen_params.append(p)
        for p in backbone.embd_norm.parameters():
            p.requires_grad = True
            af_unfrozen_params.append(p)

        num_branch = len(backbone.branch)
        print(f"ActionFormer has {num_branch} branch blocks, unfreezing last 2")
        for blk in backbone.branch[num_branch - 2:]:
            for p in blk.parameters():
                p.requires_grad = True
                af_unfrozen_params.append(p)

        for p in actionformer_feature_extractor.channel_projector.parameters():
            p.requires_grad = True
            af_unfrozen_params.append(p)

        print(f"ActionFormer unfrozen params: {sum(p.numel() for p in af_unfrozen_params):,}")
    elif args.freeze_actionformer and not args.skip_temporal:
        print("ABLATION: ActionFormer fully frozen (no selective unfreeze)")
    else:
        print("ActionFormer skipped — no params to unfreeze")

    pose_decoder.apply(initialize_gelu_weights)
    spatial_joint_transformer.apply(initialize_gelu_weights)
    heatmap_embedding.apply(initialize_relu_weights)

    print("Loading pre-trained modules for fine-tuning...")

    # Encoder: only load if temporal stream is active
    if not args.skip_temporal:
        if os.path.exists(args.encoder_path):
            print(f"Loading encoder from {args.encoder_path}")
            encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        else:
            raise FileNotFoundError(f"Encoder not found at {args.encoder_path}")

    # Pose decoder: always load (architecture must match — PoseDecoder/ConcatFusion/MLP)
    if os.path.exists(args.decoder_path):
        print(f"Loading pose decoder from {args.decoder_path}")
        pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    else:
        raise FileNotFoundError(f"Pose decoder not found at {args.decoder_path}")

    # Spatial modules: only load if spatial stream is active
    if not args.skip_spatial:
        if os.path.exists(args.heatmap_path):
            print(f"Loading heatmap embedding from {args.heatmap_path}")
            heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))
        else:
            raise FileNotFoundError(f"Heatmap embedding not found at {args.heatmap_path}")

        if not args.skip_sjt:
            if os.path.exists(args.spatial_transformer_path):
                print(f"Loading spatial joint transformer from {args.spatial_transformer_path}")
                spatial_joint_transformer.load_state_dict(
                    torch.load(args.spatial_transformer_path, map_location=device)
                )
            else:
                raise FileNotFoundError(f"Spatial joint transformer not found at {args.spatial_transformer_path}")

    print("All fine-tuning modules initialized and loaded")

    limb_loss_func = LossFuncLimb().to(device)
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim().to(device)

    enc_params = 0 if encoder is None else sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("encoder trainable params:", enc_params)

    af_param_ids = {id(p) for p in af_unfrozen_params}
    trainable_modules = [pose_decoder]
    if not args.skip_temporal:
        trainable_modules.append(encoder)
    if not args.skip_spatial:
        trainable_modules.append(heatmap_embedding)
        if not args.skip_sjt:
            trainable_modules.append(spatial_joint_transformer)
    all_other_params = [p for m in trainable_modules for p in m.parameters()
                        if p.requires_grad and id(p) not in af_param_ids]

    all_trainable_params = all_other_params + af_unfrozen_params

    if af_unfrozen_params:
        optimizer = torch.optim.AdamW([
            {'params': all_other_params, 'lr': args.learning_rate},
            {'params': af_unfrozen_params, 'lr': args.learning_rate * 0.1}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            all_other_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    amp_enabled = torch.cuda.is_available()
    scaler = GradScaler(device='cuda', enabled=amp_enabled)

    print(f"Checkpoint directory: {model_dir}")
    total_step = len(data_loader)

    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
        if encoder is not None:
            encoder.train()
        pose_decoder.train()
        heatmap_embedding.train()
        spatial_joint_transformer.train()

        train_mpjpe_losses = []
        train_cos_losses = []
        train_bone_losses = []
        train_total_losses = []

        for i, batch in enumerate(data_loader):
            print("Printing iteration number:", i)
            images = batch['input_rgb'].to(device)
            B, T, _, H_img, W_img = images.shape
            H_hm, W_hm = 64, 64
            gt_egoposes = batch['gt_local_pose'].to(device)

            if args.skip_temporal:
                motion_features = torch.zeros(B, T, 384, device=device)
            else:
                with autocast(device_type='cuda', enabled=amp_enabled):
                    motion_features = encoder(images)
                motion_features = motion_features.float()

            if args.skip_spatial:
                spatial_joint_features = torch.zeros(B, T, 15, args.hm_embed_dim, device=device)
            else:
                with torch.no_grad():
                    all_images_flat = images.view(-1, 3, H_img, W_img)
                    all_heatmaps = net_heatmap(all_images_flat)
                    all_heatmaps = torch.sigmoid(all_heatmaps)
                heatmaps = all_heatmaps.detach().view(B, T, 15, H_hm, W_hm)
                heatmaps.requires_grad_(True)

                heatmap_features = heatmap_embedding(heatmaps)
                if args.skip_sjt:
                    spatial_joint_features = heatmap_features
                else:
                    spatial_joint_features = spatial_joint_transformer(heatmap_features)

                if i == 0:
                    print(
                        f"DEBUG: heatmap_features shape={heatmap_features.shape}, "
                        f"dtype={heatmap_features.dtype}"
                    )
                    print(f"DEBUG: spatial_joint_features shape={spatial_joint_features.shape}")

            with autocast(device_type='cuda', enabled=amp_enabled):
                pose_logits = pose_decoder(spatial_joint_features, motion_features)
                final = pose_logits.view(B, T, num_joints, 3).float()

                final_reshaped = final.reshape(B * T, num_joints, 3)
                gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)

                mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
                bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
                cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)

                final_loss = (
                    opt.lambda_mpjpe * mpjpe_loss +
                    opt.lambda_cos_sim * cos_loss +
                    opt.lambda_bone_length * bone_length_loss
                )

            train_mpjpe_losses.append(mpjpe_loss.item())
            train_cos_losses.append(cos_loss.item())
            train_bone_losses.append(bone_length_loss.item())
            train_total_losses.append(final_loss.item())

            print(
                f"MPJPE: {mpjpe_loss:.4f}, "
                f"Cos: {cos_loss.item():.4f}, Bone: {bone_length_loss.item():.4f}"
            )

            optimizer.zero_grad()
            scaler.scale(final_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable_params, args.clip_value)

            if i % args.log_step == 0:
                if encoder is not None:
                    print(f'Encoder grad norm: {get_grad_norm(encoder, "Encoder"):.6f}')
                print(f'Decoder grad norm: {get_grad_norm(pose_decoder, "Decoder"):.6f}')
                print(f'Heatmap grad norm: {get_grad_norm(heatmap_embedding, "Heatmap"):.6f}')
                print(f'Spatial grad norm: {get_grad_norm(spatial_joint_transformer, "Spatial"):.6f}')

            scaler.step(optimizer)
            scaler.update()

            if i % args.log_step == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},'.format(
                        epoch, args.num_epochs, i, total_step, final_loss.item()
                    )
                )

        train_avg_mpjpe = np.mean(train_mpjpe_losses)
        train_avg_cos = np.mean(train_cos_losses)
        train_avg_bone = np.mean(train_bone_losses)
        train_avg_total = np.mean(train_total_losses)

        if train_avg_total < best_loss:
            best_loss = train_avg_total
            best_epoch = epoch

            print(f"New best model! Loss: {best_loss:.4f} (Epoch {best_epoch})")
            torch.save(pose_decoder.state_dict(), os.path.join(model_dir, 'pose-decoder-best.ckpt'))
            if not args.skip_temporal:
                torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder-best.ckpt'))
            if not args.skip_spatial:
                torch.save(heatmap_embedding.state_dict(),
                           os.path.join(model_dir, 'heatmap_embedding-best.ckpt'))
                if not args.skip_sjt:
                    torch.save(spatial_joint_transformer.state_dict(),
                               os.path.join(model_dir, 'spatial_transformer-best.ckpt'))
            print(f"Saved best checkpoints to {model_dir}")

        if (epoch + 1) % args.save_interval == 0:
            print(f"Saving periodic checkpoints for epoch {epoch + 1} to {model_dir}")
            torch.save(pose_decoder.state_dict(),
                       os.path.join(model_dir, f'pose-decoder-{epoch + 1:03d}.ckpt'))
            if not args.skip_temporal:
                torch.save(encoder.state_dict(),
                           os.path.join(model_dir, f'encoder-{epoch + 1:03d}.ckpt'))
            if not args.skip_spatial:
                torch.save(heatmap_embedding.state_dict(),
                           os.path.join(model_dir, f'heatmap_embedding-{epoch + 1:03d}.ckpt'))
                if not args.skip_sjt:
                    torch.save(spatial_joint_transformer.state_dict(),
                               os.path.join(model_dir, f'spatial_transformer-{epoch + 1:03d}.ckpt'))
            print(f"Saved periodic checkpoints for epoch {epoch + 1} to {model_dir}")

        print(f"=== Epoch {epoch} Summary ===")
        print(
            f"Train - MPJPE: {train_avg_mpjpe:.4f}, "
            f"Cos: {train_avg_cos:.4f}, Bone: {train_avg_bone:.4f}, Total: {train_avg_total:.4f}"
        )
        print(f"Best loss so far: {best_loss:.4f} (Epoch {best_epoch})")

        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print("New learning rate:", new_lr)

        with open(loss_log_path, 'a') as log_f:
            log_f.write(
                f"{epoch},"
                f"{train_avg_mpjpe:.4f},{train_avg_cos:.4f},"
                f"{train_avg_bone:.4f},{train_avg_total:.4f},{best_loss:.4f}\n"
            )

    print("\nTraining completed!")
    print(f"Best model was from epoch {best_epoch} with loss {best_loss:.4f}")
    print("Best model checkpoints saved as:")
    print("  • pose-decoder-best.ckpt")
    if not args.skip_temporal:
        print("  • encoder-best.ckpt")
    print("  • heatmap_embedding-best.ckpt")
    print("  • spatial_transformer-best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--model_path', type=str, default='./utils/trained_finetuned_sceneego',
                        help='path for saving fine-tuned models')
    parser.add_argument('--heatmap_trained_path', type=str,
                        default='/data/My_Backup/ag-egopose-ckpt/trained_heatmaps/bce_combined/heatmap_best.ckpt',
                        help='path for trained 2D heatmap')
    parser.add_argument('--heatmap_backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='backbone for 2D heatmap network (default: convnext_tiny)')

    parser.add_argument('--encoder_path', type=str,
                        default='/data/My_Backup/ag-egopose-ckpt/encoder-best.ckpt',
                        help='path for pre-trained encoder')
    parser.add_argument('--decoder_path', type=str,
                        default='/data/My_Backup/ag-egopose-ckpt/pose-decoder-best.ckpt',
                        help='path for pre-trained decoder')
    parser.add_argument('--heatmap_path', type=str,
                        default='/data/My_Backup/ag-egopose-ckpt/heatmap_embedding-best.ckpt',
                        help='path for pre-trained heatmap embedding')
    parser.add_argument('--spatial_transformer_path', type=str,
                        default='/data/My_Backup/ag-egopose-ckpt/spatial_transformer-best.ckpt',
                        help='path for pre-trained spatial transformer')

    parser.add_argument('--image_dir', type=str, default='/data/My_Backup/UnrealEgo/scripts/data/UnrealEgoData',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='D:/Dataset/EgoPW_dataset/EgoPW_dataset_release',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')

    parser.add_argument('--embed_feature_dim', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hm_embed_dim', type=int, default=128,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--seq_length', type=int, default=64,
                        help='length of the pose/video sequences')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lambda_heatmap', type=float, default=0.1,
                        help='weight for heatmap loss')

    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=100,
                        help='step size for saving trained models')
    parser.add_argument('--save_interval', type=int, default=2,
                        help='save checkpoint every N epochs')

    # Ablation flags (must match train.py)
    parser.add_argument('--skip_temporal', action='store_true',
                        help='ablation: skip temporal stream (DINOv2 + ActionFormer)')
    parser.add_argument('--skip_spatial', action='store_true',
                        help='ablation: skip spatial stream (heatmaps/SJT)')
    parser.add_argument('--skip_sjt', action='store_true',
                        help='ablation: skip SpatialJointTransformer')
    parser.add_argument('--skip_mcjq', action='store_true',
                        help='ablation: skip Motion-Conditioned Joint Queries')
    parser.add_argument('--concat_fusion', action='store_true',
                        help='ablation: concat + self-attention fusion')
    parser.add_argument('--mlp_decoder', action='store_true',
                        help='ablation: simple MLP decoder')
    parser.add_argument('--freeze_actionformer', action='store_true',
                        help='ablation: keep ActionFormer fully frozen')
    parser.add_argument('--random_actionformer', action='store_true',
                        help='ablation: randomly initialized ActionFormer')

    args = parser.parse_args()
    print(args)
    main(args)
