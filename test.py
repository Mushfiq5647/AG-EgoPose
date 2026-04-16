# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
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

torch.manual_seed(7)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_joints = 15


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def run_forward(net_heatmap, encoder, heatmap_embedding,
                spatial_joint_transformer, pose_decoder, images):
    """Single forward pass through the full pipeline."""
    B, T, _, H_img, W_img = images.shape
    all_images_flat = images.view(-1, 3, H_img, W_img)
    all_heatmaps = torch.sigmoid(net_heatmap(all_images_flat))
    heatmaps = all_heatmaps.view(B, T, 15, 64, 64)

    motion_features = encoder(images)
    heatmap_features = heatmap_embedding(heatmaps)
    spatial_joint_features = spatial_joint_transformer(heatmap_features)
    pose_logits = pose_decoder(spatial_joint_features, motion_features)
    return pose_logits


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn, n_warmup, n_runs):
    """Warmup then time a zero-arg callable. Returns (mean_s, std_s)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = fn()
        _sync()
        times = []
        for _ in range(n_runs):
            _sync()
            t0 = time.perf_counter()
            _ = fn()
            _sync()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def report_params(models):
    net_heatmap, encoder, heatmap_embedding, sjt, pose_decoder = models
    counts = {
        'net_heatmap (frozen)':      count_params(net_heatmap),
        'encoder (DINOv2+AF)':       count_params(encoder),
        'heatmap_embedding':         count_params(heatmap_embedding),
        'spatial_joint_transformer': count_params(sjt),
        'pose_decoder':              count_params(pose_decoder),
    }
    total = sum(counts.values())

    print("\n=== Parameter Counts ===")
    for name, n in counts.items():
        print(f"  {name:30s}: {n:>12,}  ({n/1e6:6.2f} M)")
    print(f"  {'TOTAL':30s}: {total:>12,}  ({total/1e6:6.2f} M)")


def benchmark(models, sample_images, n_warmup=10, n_runs=50, tag=""):
    """FLOPs + full-pipeline latency + per-module breakdown, for one batch shape."""
    net_heatmap, encoder, heatmap_embedding, sjt, pose_decoder = models
    B, T, _, H, W = sample_images.shape

    header = f" Benchmark [B={B}, T={T}] {tag} ".center(62, "=")
    print(f"\n{header}")

    # ── FLOPs ────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                run_forward(*models, sample_images)
        total_flops = fc.get_total_flops()
        print(f"FLOPs / batch : {total_flops/1e9:10.3f} G   "
              f"(/frame : {total_flops/(B*T)/1e9:.3f} G)")
    except Exception as e:
        print(f"FLOP counting failed: {e}")

    # ── Full-pipeline latency / FPS ──────────────────────────────────
    mean_s, std_s = _time_fn(lambda: run_forward(*models, sample_images), n_warmup, n_runs)
    latency_ms = mean_s * 1000
    per_frame_ms = latency_ms / (B * T)
    fps_offline = (B * T) / mean_s          # sequence/batch throughput
    fps_online = 1.0 / mean_s                # windows per second (one prediction per forward)

    print(f"\nFull pipeline ({n_runs} runs + {n_warmup} warmup):")
    print(f"  Latency / batch     : {latency_ms:8.2f} ± {std_s*1000:.2f} ms")
    print(f"  Latency / frame     : {per_frame_ms:8.2f} ms")
    if B == 1:
        print(f"  Offline FPS (T/lat) : {fps_offline:8.2f}  ← standard paper number")
        print(f"  Online  FPS (1/lat) : {fps_online:8.2f}  ← windows/second (real-time)")
    else:
        print(f"  Throughput (frames) : {fps_offline:8.2f}  ← batch-amortized, NOT standard paper FPS")

    # ── Per-module latency breakdown ─────────────────────────────────
    # Precompute intermediate tensors so each stage times only its own work
    with torch.no_grad():
        all_images_flat = sample_images.view(-1, 3, H, W)
        all_heatmaps = torch.sigmoid(net_heatmap(all_images_flat))
        heatmaps = all_heatmaps.view(B, T, 15, 64, 64)
        motion_features = encoder(sample_images)
        heatmap_features = heatmap_embedding(heatmaps)
        spatial_joint_features = sjt(heatmap_features)

    stages = [
        ('net_heatmap (frozen)',      lambda: torch.sigmoid(net_heatmap(all_images_flat))),
        ('encoder (DINOv2+AF)',       lambda: encoder(sample_images)),
        ('heatmap_embedding',         lambda: heatmap_embedding(heatmaps)),
        ('spatial_joint_transformer', lambda: sjt(heatmap_features)),
        ('pose_decoder',              lambda: pose_decoder(spatial_joint_features, motion_features)),
    ]

    print(f"\nPer-module latency (batch shape B={B}, T={T}):")
    stage_total = 0.0
    stage_results = []
    for name, fn in stages:
        mean_s_st, std_s_st = _time_fn(fn, n_warmup=5, n_runs=20)
        stage_total += mean_s_st
        stage_results.append((name, mean_s_st))
        print(f"  {name:30s}: {mean_s_st*1000:8.2f} ± {std_s_st*1000:.2f} ms")

    print(f"  {'(sum of stages)':30s}: {stage_total*1000:8.2f} ms "
          f"(vs pipeline {latency_ms:.2f} ms)")
    print(f"\nRelative share of pipeline latency:")
    for name, s in stage_results:
        pct = 100.0 * s / max(stage_total, 1e-9)
        bar = "█" * int(pct / 2)
        print(f"  {name:30s}: {pct:5.1f}%  {bar}")


def main(args):
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    opt = TestOptions().parse()
    test_loader = dataloader_full(opt, transform, mode='test')
    print(f"Test dataset: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    for p in actionformer_feature_extractor.parameters():
        p.requires_grad = False

    # ── Models ───────────────────────────────────────────────────────
    net_heatmap = HeatMap_Network(opt, model_name=args.heatmap_backbone).to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=args.hm_embed_dim, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)

    # ── Load checkpoints ─────────────────────────────────────────────
    if os.path.exists(args.heatmap_trained_path):
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
    else:
        print(f"WARN: heatmap checkpoint missing: {args.heatmap_trained_path}")
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))
    spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))

    for m in (net_heatmap, encoder, pose_decoder, heatmap_embedding, spatial_joint_transformer):
        m.eval()
    for p in net_heatmap.parameters():
        p.requires_grad = False

    models = (net_heatmap, encoder, heatmap_embedding, spatial_joint_transformer, pose_decoder)

    # ── Benchmark ────────────────────────────────────────────────────
    sample_batch = next(iter(test_loader))
    sample_images = sample_batch['input_rgb'].to(device)
    print(f"\nBenchmark sample: {sample_images.shape}")

    report_params(models)
    benchmark(models, sample_images[:1], n_warmup=args.warmup, n_runs=args.bench_runs,
              tag="(single-sequence, paper FPS)")
    if sample_images.shape[0] > 1:
        benchmark(models, sample_images, n_warmup=args.warmup, n_runs=args.bench_runs,
                  tag="(batch throughput)")

    if args.bench_only:
        return

    # ── Evaluation ───────────────────────────────────────────────────
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    total_mpjpe = 0.0
    total_pa_mpjpe = 0.0
    total_samples = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['input_rgb'].to(device)
            gt_egoposes = batch['gt_local_pose'].to(device)
            B, T = images.shape[:2]

            pose_logits = run_forward(*models, images)
            final = pose_logits.view(B, T, num_joints, 3)

            final_reshaped = final.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)

            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            S1_hat = batch_compute_similarity_transform_torch(final_reshaped, gt_reshaped)
            pa_mpjpe = mpjpe_loss_func(S1_hat, gt_reshaped)

            total_mpjpe += mpjpe_loss.item() * B * T
            total_pa_mpjpe += pa_mpjpe.item() * B * T
            total_samples += B * T

            if (i + 1) % args.log_step == 0:
                print(f'  [{i+1}/{len(test_loader)}] running MPJPE={total_mpjpe/total_samples:.4f}')

    avg_mpjpe = total_mpjpe / total_samples
    avg_pa_mpjpe = total_pa_mpjpe / total_samples

    print(f'\n=== Final Results ===')
    print(f'MPJPE    : {avg_mpjpe:.4f}')
    print(f'PA-MPJPE : {avg_pa_mpjpe:.4f}')
    print(f'Samples  : {total_samples}')

    with open('test_results.txt', 'w') as f:
        f.write(f"MPJPE: {avg_mpjpe:.4f}\n")
        f.write(f"PA-MPJPE: {avg_pa_mpjpe:.4f}\n")
        f.write(f"Samples: {total_samples}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    parser.add_argument('--encoder_path', type=str, required=True)
    parser.add_argument('--decoder_path', type=str, required=True)
    parser.add_argument('--heatmap_trained_path', type=str, required=True)
    parser.add_argument('--heatmap_path', type=str, required=True)
    parser.add_argument('--spatial_transformer_path', type=str, required=True)
    parser.add_argument('--heatmap_backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'resnet18', 'resnet34', 'resnet50', 'resnet101'])

    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect')

    parser.add_argument('--embed_feature_dim', type=int, default=256)
    parser.add_argument('--hm_embed_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--log_step', type=int, default=20)

    # Benchmarking
    parser.add_argument('--warmup', type=int, default=10, help='warmup runs before latency timing')
    parser.add_argument('--bench_runs', type=int, default=50, help='timed runs for latency')
    parser.add_argument('--bench_only', action='store_true', help='run benchmark and skip full evaluation')

    args = parser.parse_args()
    main(args)
