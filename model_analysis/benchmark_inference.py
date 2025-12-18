"""
Benchmark script to measure inference latency and prove real-time deployability.
Based on test.py structure with added latency measurements.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from torchvision import transforms

from utils.action_recognition import initialize_actionformer
from options.test_options import TestOptions
from utils.data_loader import dataloader_full
from utils.cross_attention_model import HeatmapToJointFeatures
from utils.cross_attention_model import SpatialJointTransformer
from utils.cross_attention_model import PoseDecoder
from utils.loss import LossFuncMPJPE
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.util import batch_compute_similarity_transform_torch
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


def main(args):
    # -----------------------------
    # 1. Init ActionFormer backbone
    # -----------------------------
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)

    # Image preprocessing (same as train.py)
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    opt = TestOptions().parse()
    test_loader = dataloader_full(opt, transform, mode='test')
    print("Test Data Loading complete", len(test_loader))
    print("Total test dataset", len(test_loader.dataset))

    print("Freezing ActionFormer...")
    for param in actionformer_feature_extractor.parameters():
        param.requires_grad = False
    print("ActionFormer frozen")

    # -----------------------------
    # 2. Initialize models
    # -----------------------------
    net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
    heatmap_embedding = HeatmapToJointFeatures(
        heatmap_size=64, feature_dim=128, method='conv_pool'
    ).to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(
        args.hm_embed_dim, num_heads=4, num_layers=3
    ).to(device)
    pose_decoder = PoseDecoder(
        args.hm_embed_dim, num_heads=4, num_layers=3
    ).to(device)

    # Load pre-trained 2D heatmap network
    if os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        net_heatmap.eval()
        for param in net_heatmap.parameters():
            param.requires_grad = False
        print("2D heatmap network loaded and frozen")
    else:
        print(f"Warning: Pre-trained heatmap model not found at {args.heatmap_trained_path}")

    # Load trained models
    print(f"Loading encoder from {args.encoder_path}")
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

    print(f"Loading pose decoder from {args.decoder_path}")
    pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    print(f"Loading heatmap embedding from {args.heatmap_path}")
    heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))

    print(f"Loading spatial transformer from {args.spatial_transformer_path}")
    spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))

    # Set all models to evaluation mode
    net_heatmap.eval()
    encoder.eval()
    pose_decoder.eval()
    heatmap_embedding.eval()
    spatial_joint_transformer.eval()

    print("All models loaded and set to evaluation mode")

    # -----------------------------
    # 3. Loss + metrics trackers
    # -----------------------------
    mpjpe_loss_func = LossFuncMPJPE().to(device)

    # Joint names (from loss.py)
    joint_names = ["Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
                   "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
                   "Left_knee", "Left_ankle", "Left_foot"]
    
    total_mpjpe = 0.0
    total_procrustes_error = 0.0
    total_samples = 0

    # Per-joint error accumulators
    per_joint_mpjpe_sum = torch.zeros(num_joints).to(device)  # Sum of errors per joint
    per_joint_pa_mpjpe_sum = torch.zeros(num_joints).to(device)  # Sum of PA-MPJPE per joint
    per_joint_sample_count = torch.zeros(num_joints).to(device)  # Sample count per joint (should be same for all)

    # Latency tracking
    all_batch_latencies = []           # ms per batch (B sequences)
    all_seq_latencies = []             # ms per sequence
    seq_length = None                  # will be set from first batch
    batch_size_first = None

    print("Starting evaluation with latency measurements...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['input_rgb'].to(device)   # [B, T, 3, H, W]
            B, T, _, H_img, W_img = images.shape
            if seq_length is None:
                seq_length = T
            if batch_size_first is None:
                batch_size_first = B

            H_hm, W_hm = 64, 64
            gt_egoposes = batch['gt_local_pose'].to(device)  # [B, T, J, 3]

            # -----------------------------
            # 4. Measure forward latency
            # -----------------------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # Heatmap branch
            all_images_flat = images.view(-1, 3, H_img, W_img)              # [B*T, 3, H, W]
            all_heatmaps = net_heatmap(all_images_flat)                     # [B*T, J, H_hm, W_hm]
            all_heatmaps = torch.sigmoid(all_heatmaps)                      # convert logits → [0,1]
            heatmaps = all_heatmaps.view(B, T, num_joints, H_hm, W_hm)      # [B, T, J, H_hm, W_hm]

            # ActionFormer + fusion
            motion_features = encoder(images)                               # [B, T, D_m]
            heatmap_features = heatmap_embedding(heatmaps)                  # [B, T, J, D_h]
            spatial_joint_features = spatial_joint_transformer(heatmap_features)  # (not used directly, but ok)
            pose_logits = pose_decoder(heatmap_features, motion_features)   # [B, T, J*3]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            batch_latency_ms = (end_time - start_time) * 1000.0
            latency_per_sequence_ms = batch_latency_ms / B

            all_batch_latencies.append(batch_latency_ms)
            all_seq_latencies.append(latency_per_sequence_ms)

            # -----------------------------
            # 5. Accuracy metrics (not timed)
            # -----------------------------
            final = pose_logits.view(B, T, num_joints, 3)
            final_reshaped = final.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)

            # Overall MPJPE and PA-MPJPE
            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            S1_hat = batch_compute_similarity_transform_torch(final_reshaped, gt_reshaped)
            pa_mpjpe = mpjpe_loss_func(S1_hat, gt_reshaped)

            # Per-joint MPJPE: calculate distance per joint, then average over samples
            # distance shape: [B*T, num_joints] - Euclidean distance for each sample-joint pair
            mpjpe_per_joint = torch.linalg.norm(gt_reshaped - final_reshaped, dim=-1)  # [B*T, num_joints]
            mpjpe_per_joint_mean = torch.mean(mpjpe_per_joint, dim=0)  # [num_joints] - average over samples
            
            # Per-joint PA-MPJPE: same calculation but with Procrustes-aligned predictions
            pa_mpjpe_per_joint = torch.linalg.norm(gt_reshaped - S1_hat, dim=-1)  # [B*T, num_joints]
            pa_mpjpe_per_joint_mean = torch.mean(pa_mpjpe_per_joint, dim=0)  # [num_joints]

            # Accumulate per-joint errors
            # mpjpe_per_joint_mean is already averaged over (B*T) samples, so we accumulate weighted by batch size
            per_joint_mpjpe_sum += mpjpe_per_joint_mean * (B * T)
            per_joint_pa_mpjpe_sum += pa_mpjpe_per_joint_mean * (B * T)
            per_joint_sample_count += (B * T)  # Same count for all joints

            print(f'Batch [{i + 1}/{len(test_loader)}], MPJPE: {mpjpe_loss:.4f}, PA-MPJPE: {pa_mpjpe:.4f}')
            print(f'  Latency: {batch_latency_ms:.2f} ms (batch of {B} sequences), '
                  f'{latency_per_sequence_ms:.2f} ms/sequence ({T} frames)')

            total_mpjpe += mpjpe_loss.item() * B * T
            total_procrustes_error += pa_mpjpe.item() * B * T
            total_samples += B * T

        # -----------------------------
        # 6. Final averages
        # -----------------------------
        avg_mpjpe = total_mpjpe / total_samples
        avg_pa_mpjpe = total_procrustes_error / total_samples
        
        # Per-joint averages
        per_joint_mpjpe_avg = (per_joint_mpjpe_sum / per_joint_sample_count[0]).cpu().numpy()
        per_joint_pa_mpjpe_avg = (per_joint_pa_mpjpe_sum / per_joint_sample_count[0]).cpu().numpy()

        avg_batch_latency_ms = float(np.mean(all_batch_latencies)) if all_batch_latencies else 0.0
        std_batch_latency_ms = float(np.std(all_batch_latencies)) if all_batch_latencies else 0.0
        min_batch_latency_ms = float(np.min(all_batch_latencies)) if all_batch_latencies else 0.0
        max_batch_latency_ms = float(np.max(all_batch_latencies)) if all_batch_latencies else 0.0

        avg_seq_latency_ms = float(np.mean(all_seq_latencies)) if all_seq_latencies else 0.0
        std_seq_latency_ms = float(np.std(all_seq_latencies)) if all_seq_latencies else 0.0

        # Offline throughput (bulk-processing FPS) - *not* real-time FPS
        avg_frames_per_batch = total_samples / len(all_batch_latencies) if all_batch_latencies else 0.0
        offline_fps = (avg_frames_per_batch * 1000.0) / avg_batch_latency_ms if avg_batch_latency_ms > 0 else 0.0

        # Online real-time FPS: one sequence per forward pass
        online_fps = 1000.0 / avg_seq_latency_ms if avg_seq_latency_ms > 0 else 0.0

        # Targets for real-time streaming
        required_latency_30fps = 1000.0 / 30.0    # 33.3 ms per sequence
        required_latency_60fps = 1000.0 / 60.0    # 16.7 ms per sequence
        can_do_30fps = avg_seq_latency_ms < required_latency_30fps
        can_do_60fps = avg_seq_latency_ms < required_latency_60fps

        # -----------------------------
        # 7. Print results
        # -----------------------------
        print(f'\n=== Final Evaluation Results ===')
        print(f'Average MPJPE: {avg_mpjpe:.4f}')
        print(f'Average PA-MPJPE: {avg_pa_mpjpe:.4f}')
        print(f'Total samples evaluated: {total_samples}')
        
        print(f'\n=== Per-Joint MPJPE Analysis ===')
        print(f'{"Joint Name":<20} {"MPJPE (mm)":<15} {"PA-MPJPE (mm)":<15}')
        print('-' * 50)
        for j in range(num_joints):
            print(f'{joint_names[j]:<20} {per_joint_mpjpe_avg[j]:<15.4f} {per_joint_pa_mpjpe_avg[j]:<15.4f}')

        print(f'\n=== Latency Benchmark Results ===')
        print(f'Batch size during benchmark: {batch_size_first}')
        print(f'Sequence length (T): {seq_length} frames')
        print(f'Batch Latency (B sequences): {avg_batch_latency_ms:.2f} ± {std_batch_latency_ms:.2f} ms')
        print(f'  Min/Max batch latency: {min_batch_latency_ms:.2f} / {max_batch_latency_ms:.2f} ms')
        print(f'Latency per Sequence: {avg_seq_latency_ms:.2f} ± {std_seq_latency_ms:.2f} ms')
        print(f'  Online FPS (1 stream): {online_fps:.2f} fps')

        print(f'\nOffline Throughput (bulk processing, NOT real-time):')
        print(f'  Avg frames per batch: {avg_frames_per_batch:.2f}')
        print(f'  Offline throughput:   {offline_fps:.2f} frames/s')

        print(f'\nReal-time Streaming Capability (per sequence):')
        print(f'  Need < {required_latency_30fps:.1f} ms for 30 FPS, '
              f'< {required_latency_60fps:.1f} ms for 60 FPS')
        print(f'  Can achieve 30 FPS: {"✓ YES" if can_do_30fps else "✗ NO"} '
              f'({avg_seq_latency_ms:.2f} ms/sequence)')
        print(f'  Can achieve 60 FPS: {"✓ YES" if can_do_60fps else "✗ NO"} '
              f'({avg_seq_latency_ms:.2f} ms/sequence)')

        # -----------------------------
        # 8. Save results to file
        # -----------------------------
        results_file = 'benchmark_results.txt'
        with open(results_file, 'w') as f:
            f.write("Inference Benchmark Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("ACCURACY METRICS:\n")
            f.write(f"Average MPJPE: {avg_mpjpe:.4f}\n")
            f.write(f"Average PA-MPJPE: {avg_pa_mpjpe:.4f}\n")
            f.write(f"Total samples evaluated: {total_samples}\n\n")
            f.write("PER-JOINT MPJPE ANALYSIS:\n")
            f.write(f"{'Joint Name':<20} {'MPJPE (mm)':<15} {'PA-MPJPE (mm)':<15}\n")
            f.write("-" * 50 + "\n")
            for j in range(num_joints):
                f.write(f"{joint_names[j]:<20} {per_joint_mpjpe_avg[j]:<15.4f} {per_joint_pa_mpjpe_avg[j]:<15.4f}\n")
            f.write("\n")
            f.write("LATENCY METRICS:\n")
            f.write(f"Batch size during benchmark: {batch_size_first}\n")
            f.write(f"Sequence length (T): {seq_length} frames\n")
            f.write(f"Batch Latency (B sequences): {avg_batch_latency_ms:.2f} ± {std_batch_latency_ms:.2f} ms\n")
            f.write(f"  Min/Max batch latency: {min_batch_latency_ms:.2f} / {max_batch_latency_ms:.2f} ms\n")
            f.write(f"Latency per Sequence: {avg_seq_latency_ms:.2f} ± {std_seq_latency_ms:.2f} ms\n")
            f.write(f"Online FPS (1 stream): {online_fps:.2f} fps\n\n")
            f.write("OFFLINE THROUGHPUT (NOT real-time FPS):\n")
            f.write(f"Avg frames per batch: {avg_frames_per_batch:.2f}\n")
            f.write(f"Offline throughput:   {offline_fps:.2f} frames/s\n\n")
            f.write("REAL-TIME STREAMING CAPABILITY:\n")
            f.write(f"Required for 30 FPS: < {required_latency_30fps:.1f} ms/sequence\n")
            f.write(f"Required for 60 FPS: < {required_latency_60fps:.1f} ms/sequence\n")
            f.write(f"Can achieve 30 FPS: {can_do_30fps} ({avg_seq_latency_ms:.2f} ms/sequence)\n")
            f.write(f"Can achieve 60 FPS: {can_do_60fps} ({avg_seq_latency_ms:.2f} ms/sequence)\n")

        print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark inference latency for real-time deployment')
    print(sys.argv)

    # Model and data paths (same as test.py)
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
    parser.add_argument('--heatmap_path', type=str, required=True, help='path for trained heatmap embedding')
    parser.add_argument('--spatial_transformer_path', type=str, required=True, help='path for trained spatial transformer')

    # Directories (same as test.py)
    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for homography')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for OpenPose JSON files')

    # Model parameters (same as test.py)
    parser.add_argument('--embed_feature_dim', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hm_embed_dim', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--seq_length', type=int, default=32, help='length of the pose/video sequences')

    # Other settings (same as test.py)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=256, help='size for randomly cropping images')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')

    args = parser.parse_args()
    main(args)
