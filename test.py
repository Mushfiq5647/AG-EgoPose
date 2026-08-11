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
from utils.cross_attention_model import PoseDecoder, ConcatFusionDecoder
from utils.model import MLPPoseDecoder
from utils.loss import LossFuncMPJPE
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.util import batch_compute_similarity_transform_torch

torch.manual_seed(7)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_joints = 15
JOINT_NAMES = [
    "Neck",
    "Right_shoulder",
    "Right_elbow",
    "Right_wrist",
    "Left_shoulder",
    "Left_elbow",
    "Left_wrist",
    "Right_hip",
    "Right_knee",
    "Right_ankle",
    "Right_foot",
    "Left_hip",
    "Left_knee",
    "Left_ankle",
    "Left_foot",
]


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def configure_actionformer_trainable(actionformer_feature_extractor, skip_temporal=False, freeze_actionformer=False):
    """Mirror train.py's selective ActionFormer unfreeze so param reporting matches training."""
    af_unfrozen_params = []
    if skip_temporal or actionformer_feature_extractor is None:
        print("ActionFormer skipped — no params to unfreeze")
        return af_unfrozen_params

    if freeze_actionformer:
        print("ABLATION: ActionFormer fully frozen (no selective unfreeze)")
        return af_unfrozen_params

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
    return af_unfrozen_params


def run_forward(net_heatmap, encoder, heatmap_embedding,
                spatial_joint_transformer, pose_decoder, images,
                skip_temporal=False, skip_spatial=False, skip_sjt=False,
                hm_embed_dim=128):
    """Single forward pass through the full pipeline."""
    B, T, _, H_img, W_img = images.shape

    if skip_temporal:
        motion_features = torch.zeros(B, T, 384, device=images.device)
    else:
        motion_features = encoder(images)

    if skip_spatial:
        spatial_joint_features = torch.zeros(B, T, 15, hm_embed_dim, device=images.device)
    else:
        all_images_flat = images.view(-1, 3, H_img, W_img)
        all_heatmaps = torch.sigmoid(net_heatmap(all_images_flat))
        heatmaps = all_heatmaps.view(B, T, 15, 64, 64)
        heatmap_features = heatmap_embedding(heatmaps)
        if skip_sjt:
            spatial_joint_features = heatmap_features
        else:
            spatial_joint_features = spatial_joint_transformer(heatmap_features)

    pose_logits = pose_decoder(spatial_joint_features, motion_features)
    return pose_logits


def run_occlusion_analysis(models, test_loader, ablation_kwargs, tag="",
                           bins=(0.0, 0.2, 0.5, 1.01)):
    """Bucket per-joint MPJPE and PA-MPJPE by heatmap-peak visibility (R4 W2).

    For each frame/joint, the max of the sigmoid heatmap is a visibility
    proxy in [0,1]. PA-MPJPE requires a whole-pose Procrustes alignment
    (same convention as the main eval loop), so we align per-sequence
    first and then bucket the resulting per-joint errors by visibility.
    Writes a CSV with both metrics per bucket.
    """
    net_heatmap, encoder, heatmap_embedding, sjt, pose_decoder = models
    n_bins = len(bins) - 1
    err_sum = torch.zeros(n_bins, device=device)
    pa_err_sum = torch.zeros(n_bins, device=device)
    cnt = torch.zeros(n_bins, device=device)
    # per-joint x per-bucket for a finer table
    perjoint_err = torch.zeros(num_joints, n_bins, device=device)
    perjoint_pa_err = torch.zeros(num_joints, n_bins, device=device)
    perjoint_cnt = torch.zeros(num_joints, n_bins, device=device)

    print(f"Occlusion analysis {tag}: bucketing per-joint MPJPE/PA-MPJPE by heatmap visibility...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['input_rgb'].to(device)
            gt = batch['gt_local_pose'].to(device)
            B, T, _, H_img, W_img = images.shape

            pose_logits = run_forward(*models, images, **ablation_kwargs)
            pred = pose_logits.view(B, T, num_joints, 3)
            gt_r = gt.view(B, T, num_joints, 3)

            pred_flat = pred.reshape(B * T, num_joints, 3)
            gt_flat = gt_r.reshape(B * T, num_joints, 3)
            S1_hat = batch_compute_similarity_transform_torch(pred_flat, gt_flat)

            joint_err = torch.linalg.norm(pred_flat - gt_flat, dim=-1)   # (B*T,J)
            joint_pa_err = torch.linalg.norm(S1_hat - gt_flat, dim=-1)   # (B*T,J)

            # Visibility proxy: peak of sigmoid heatmap per joint/frame
            flat_img = images.view(-1, 3, H_img, W_img)
            hm = torch.sigmoid(net_heatmap(flat_img)).view(B, T, num_joints, 64, 64)
            vis = hm.amax(dim=(-1, -2)).reshape(B * T, num_joints)  # (B*T,J) in [0,1]

            joint_err = joint_err.reshape(-1)
            joint_pa_err = joint_pa_err.reshape(-1)
            vis = vis.reshape(-1)
            jidx = torch.arange(num_joints, device=device).repeat(B * T)

            for b in range(n_bins):
                m = (vis >= bins[b]) & (vis < bins[b + 1])
                err_sum[b] += joint_err[m].sum()
                pa_err_sum[b] += joint_pa_err[m].sum()
                cnt[b] += m.sum()
                for j in range(num_joints):
                    mj = m & (jidx == j)
                    perjoint_err[j, b] += joint_err[mj].sum()
                    perjoint_pa_err[j, b] += joint_pa_err[mj].sum()
                    perjoint_cnt[j, b] += mj.sum()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(test_loader)}]")

    mpjpe_by_bin = (err_sum / cnt.clamp(min=1)).cpu().numpy()
    pa_mpjpe_by_bin = (pa_err_sum / cnt.clamp(min=1)).cpu().numpy()
    frac = (cnt / cnt.sum()).cpu().numpy()
    # dash-separated, no comma: commas inside a label break CSV column
    # alignment in spreadsheet viewers (header field count != data field count)
    labels = [f"{bins[b]:.1f}-{bins[b+1]:.1f}" for b in range(n_bins)]

    print("\n=== MPJPE / PA-MPJPE by visibility bucket ===")
    print(f"{'bucket':12s}  {'MPJPE':>10s}  {'PA-MPJPE':>10s}  {'%frames':>8s}")
    for b in range(n_bins):
        print(f"{labels[b]:12s}  {mpjpe_by_bin[b]:10.4f}  {pa_mpjpe_by_bin[b]:10.4f}  {100*frac[b]:8.2f}")

    out = f"occlusion_analysis{('--'+tag) if tag else ''}.csv"
    pj = (perjoint_err / perjoint_cnt.clamp(min=1)).cpu().numpy()
    pj_pa = (perjoint_pa_err / perjoint_cnt.clamp(min=1)).cpu().numpy()
    with open(out, 'w') as f:
        f.write("bucket,mpjpe,pa_mpjpe,frac_frames\n")
        for b in range(n_bins):
            f.write(f"{labels[b]},{mpjpe_by_bin[b]:.5f},{pa_mpjpe_by_bin[b]:.5f},{frac[b]:.5f}\n")
        f.write("\njoint,name," + ",".join(f"mpjpe_{l}" for l in labels) + "," \
                + ",".join(f"pa_mpjpe_{l}" for l in labels) + "\n")
        for j in range(num_joints):
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"J{j}"
            f.write(f"{j},{name}," + ",".join(f"{pj[j,b]:.5f}" for b in range(n_bins)) + "," \
                    + ",".join(f"{pj_pa[j,b]:.5f}" for b in range(n_bins)) + "\n")
    print(f"Wrote {out}")


def run_patch_occlusion(models, test_loader, ablation_kwargs, tag="",
                        patch=40, fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
                        occ_joint_idx=(7, 8, 9, 10, 11, 12, 13, 14), seed=0):
    """Causal occluder-patch robustness probe (R4).

    Unlike the confidence-bucket split (which is *observational* — it conditions
    on the spatial detector's own failures), this *actively removes* visual
    evidence: for each sequence we paste a gray occluder patch over the 2D
    location of each lower-body joint in a random fraction of frames, then
    re-run the pipeline. This isolates occlusion recovery: the full model can
    borrow temporal context from the unoccluded neighbouring frames, whereas
    --skip_temporal (per-frame spatial only) cannot. Masks are RNG-seeded
    independently of the model config, so the full and --skip_temporal runs see
    byte-identical occlusions and the comparison is fair.

    Joint peaks come from the clean (unmasked) heatmaps, so patch placement is
    also model-independent. Occluder value 0 == ImageNet mean == mid-gray.
    Reports overall and lower-body-only MPJPE/PA-MPJPE per occlusion fraction.
    """
    net_heatmap = models[0]
    lower = list(occ_joint_idx)
    half = patch // 2
    print(f"Patch-occlusion probe {tag}: patch={patch}px over joints "
          f"{[JOINT_NAMES[j] for j in lower]}, fracs={fracs}")

    # batch-outer / frac-inner so the clean heatmap pass runs once per batch
    accum = {f: torch.zeros(6, device=device) for f in fracs}  # err,pa,n,err_low,pa_low,n_low
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['input_rgb'].to(device)
            gt = batch['gt_local_pose'].to(device)
            B, T, _, H, W = images.shape

            # 2D joint peaks from clean heatmaps (model-independent placement)
            flat = images.view(-1, 3, H, W)
            hm = torch.sigmoid(net_heatmap(flat)).view(B, T, num_joints, 64, 64)
            peak = hm.view(B, T, num_joints, -1).argmax(-1)          # (B,T,J)
            py = (peak // 64).float() * (H / 64.0)
            px = (peak % 64).float() * (W / 64.0)
            gtf = gt.view(B, T, num_joints, 3).reshape(B * T, num_joints, 3)

            for frac in fracs:
                masked = images.clone()
                if frac > 0:
                    k = max(1, int(round(frac * T)))
                    for b in range(B):
                        g = torch.Generator()
                        g.manual_seed(seed + i * 100003 + int(frac * 1000) + b)
                        occ_t = torch.randperm(T, generator=g)[:k].tolist()
                        for t in occ_t:
                            for j in lower:
                                cy, cx = int(py[b, t, j]), int(px[b, t, j])
                                y0, y1 = max(0, cy - half), min(H, cy + half)
                                x0, x1 = max(0, cx - half), min(W, cx + half)
                                masked[b, t, :, y0:y1, x0:x1] = 0.0

                pose_logits = run_forward(*models, masked, **ablation_kwargs)
                pred = pose_logits.view(B, T, num_joints, 3).reshape(B * T, num_joints, 3)
                S1 = batch_compute_similarity_transform_torch(pred, gtf)
                je = torch.linalg.norm(pred - gtf, dim=-1)           # (B*T,J)
                pae = torch.linalg.norm(S1 - gtf, dim=-1)
                a = accum[frac]
                a[0] += je.sum(); a[1] += pae.sum(); a[2] += je.numel()
                a[3] += je[:, lower].sum(); a[4] += pae[:, lower].sum()
                a[5] += je[:, lower].numel()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(test_loader)}]")

    print(f"\n=== Patch-occlusion sweep {tag} ===")
    print(f"{'occ_frac':>8s}  {'MPJPE_all':>10s}  {'PA_all':>10s}  {'MPJPE_low':>10s}  {'PA_low':>10s}")
    rows = []
    for frac in fracs:
        a = accum[frac].cpu()
        row = (frac, (a[0] / a[2]).item(), (a[1] / a[2]).item(),
               (a[3] / a[5]).item(), (a[4] / a[5]).item())
        rows.append(row)
        print(f"{row[0]:8.2f}  {row[1]:10.4f}  {row[2]:10.4f}  {row[3]:10.4f}  {row[4]:10.4f}")

    out = f"patch_occlusion{('--'+tag) if tag else ''}.csv"
    with open(out, 'w') as f:
        f.write("occ_frac,mpjpe_all,pa_mpjpe_all,mpjpe_lower,pa_mpjpe_lower\n")
        for r in rows:
            f.write(f"{r[0]:.2f},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f},{r[4]:.5f}\n")
    print(f"Wrote {out}")


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
        'net_heatmap (frozen)':      (count_params(net_heatmap), count_trainable_params(net_heatmap)),
        'encoder (DINOv2+AF)':       (count_params(encoder), count_trainable_params(encoder)),
        'heatmap_embedding':         (count_params(heatmap_embedding), count_trainable_params(heatmap_embedding)),
        'spatial_joint_transformer': (count_params(sjt), count_trainable_params(sjt)),
        'pose_decoder':              (count_params(pose_decoder), count_trainable_params(pose_decoder)),
    }
    total = sum(total_n for total_n, _ in counts.values())
    total_trainable = sum(trainable_n for _, trainable_n in counts.values())

    print("\n=== Parameter Counts ===")
    for name, (total_n, trainable_n) in counts.items():
        print(
            f"  {name:30s}: total={total_n:>12,}  ({total_n/1e6:6.2f} M)   "
            f"trainable={trainable_n:>12,}  ({trainable_n/1e6:6.2f} M)"
        )
    print(f"  {'TOTAL':30s}: {total:>12,}  ({total/1e6:6.2f} M)")
    print(f"  {'TRAINABLE TOTAL':30s}: {total_trainable:>12,}  ({total_trainable/1e6:6.2f} M)")

    print("\n=== Encoder Breakdown ===")
    encoder_parts = {
        'encoder.dinov2 (frozen)':        encoder.dinov2,
        'encoder.bn':                     encoder.bn,
        'encoder.actionformer_total':     encoder.actionformer_model,
    }
    # Breakdown only applies to the real ActionFormer wrapper; temporal
    # baselines (tcn/lstm) have no .model.backbone — skip the sub-breakdown.
    if encoder.actionformer_model is not None and hasattr(encoder.actionformer_model, 'model'):
        encoder_parts.update({
            '  af.input_proj':            encoder.actionformer_model.input_proj,
            '  af.backbone.embd':         encoder.actionformer_model.model.backbone.embd,
            '  af.backbone.embd_norm':    encoder.actionformer_model.model.backbone.embd_norm,
            '  af.backbone.last2branch':  torch.nn.ModuleList(
                encoder.actionformer_model.model.backbone.branch[-2:]
            ),
            '  af.channel_projector':     encoder.actionformer_model.channel_projector,
        })
    for name, module in encoder_parts.items():
        print(
            f"  {name:30s}: total={count_params(module):>12,}  ({count_params(module)/1e6:6.2f} M)   "
            f"trainable={count_trainable_params(module):>12,}  ({count_trainable_params(module)/1e6:6.2f} M)"
        )


def benchmark(models, sample_images, n_warmup=10, n_runs=50, tag="",
              skip_temporal=False, skip_spatial=False, skip_sjt=False, hm_embed_dim=128):
    """FLOPs + full-pipeline latency + per-module breakdown, for one batch shape."""
    net_heatmap, encoder, heatmap_embedding, sjt, pose_decoder = models
    B, T, _, H, W = sample_images.shape
    fwd_kwargs = dict(
        skip_temporal=skip_temporal,
        skip_spatial=skip_spatial,
        skip_sjt=skip_sjt,
        hm_embed_dim=hm_embed_dim,
    )

    header = f" Benchmark [B={B}, T={T}] {tag} ".center(62, "=")
    print(f"\n{header}")

    # ── FLOPs ────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                run_forward(*models, sample_images, **fwd_kwargs)
        total_flops = fc.get_total_flops()
        print(f"FLOPs / batch : {total_flops/1e9:10.3f} G   "
              f"(/frame : {total_flops/(B*T)/1e9:.3f} G)")
    except Exception as e:
        print(f"FLOP counting failed: {e}")

    # ── Full-pipeline latency / FPS ──────────────────────────────────
    mean_s, std_s = _time_fn(lambda: run_forward(*models, sample_images, **fwd_kwargs), n_warmup, n_runs)
    latency_ms = mean_s * 1000
    per_frame_ms = latency_ms / (B * T)
    fps_offline = (B * T) / mean_s
    fps_online = 1.0 / mean_s

    print(f"\nFull pipeline ({n_runs} runs + {n_warmup} warmup):")
    print(f"  Latency / batch     : {latency_ms:8.2f} ± {std_s*1000:.2f} ms")
    print(f"  Latency / frame     : {per_frame_ms:8.2f} ms")
    if B == 1:
        print(f"  Offline FPS (T/lat) : {fps_offline:8.2f}  ← standard paper number")
        print(f"  Online  FPS (1/lat) : {fps_online:8.2f}  ← windows/second (real-time)")
    else:
        print(f"  Throughput (frames) : {fps_offline:8.2f}  ← batch-amortized, NOT standard paper FPS")

    # ── Per-module latency breakdown ─────────────────────────────────
    stages = []
    with torch.no_grad():
        if not skip_spatial:
            all_images_flat = sample_images.view(-1, 3, H, W)
            all_heatmaps = torch.sigmoid(net_heatmap(all_images_flat))
            heatmaps = all_heatmaps.view(B, T, 15, 64, 64)
            heatmap_features = heatmap_embedding(heatmaps)
            if skip_sjt:
                spatial_joint_features = heatmap_features
            else:
                spatial_joint_features = sjt(heatmap_features)
            stages += [
                ('net_heatmap (frozen)',      lambda: torch.sigmoid(net_heatmap(all_images_flat))),
                ('heatmap_embedding',         lambda: heatmap_embedding(heatmaps)),
            ]
            if not skip_sjt:
                stages.append(('spatial_joint_transformer', lambda: sjt(heatmap_features)))
        else:
            spatial_joint_features = torch.zeros(B, T, 15, hm_embed_dim, device=sample_images.device)

        if not skip_temporal:
            motion_features = encoder(sample_images)
            stages.append(('encoder (DINOv2+AF)', lambda: encoder(sample_images)))
        else:
            motion_features = torch.zeros(B, T, 384, device=sample_images.device)

    stages.append(('pose_decoder', lambda: pose_decoder(spatial_joint_features, motion_features)))

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
    if args.skip_temporal:
        actionformer_feature_extractor = None
        print("ABLATION: Skipping entire temporal stream (DINOv2 + ActionFormer)")
    elif args.temporal_baseline:
        from utils.temporal_baselines import build_temporal_baseline
        actionformer_feature_extractor = build_temporal_baseline(
            args.temporal_baseline, dim=384, causal=args.causal
        ).to(device)
        print(f"ABLATION: temporal baseline '{args.temporal_baseline}' "
              f"(causal={args.causal}) — weights loaded from encoder checkpoint")
    else:
        actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    opt = TestOptions().parse()
    test_loader = dataloader_full(opt, transform, mode='test')
    print(f"Test dataset: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    if actionformer_feature_extractor is not None:
        for p in actionformer_feature_extractor.parameters():
            p.requires_grad = False
        # Selective-unfreeze bookkeeping only applies to the real ActionFormer;
        # baselines are plain modules with weights baked into the checkpoint.
        if not args.temporal_baseline:
            configure_actionformer_trainable(
                actionformer_feature_extractor,
                skip_temporal=args.skip_temporal,
                freeze_actionformer=args.freeze_actionformer,
            )

    # ── Models ───────────────────────────────────────────────────────
    net_heatmap = HeatMap_Network(opt, model_name=args.heatmap_backbone).to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=args.hm_embed_dim, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor, skip_temporal=args.skip_temporal,
                             dino_feature=args.dino_feature).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    if args.mlp_decoder:
        pose_decoder = MLPPoseDecoder(motion_dim=384, joint_dim=args.hm_embed_dim).to(device)
        print("ABLATION: Using MLPPoseDecoder (concat + MLP, no transformer/attention)")
    elif args.concat_fusion:
        pose_decoder = ConcatFusionDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
        print("ABLATION: Using ConcatFusionDecoder (concat + self-attention)")
    else:
        pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3,
                                   use_mcjq=not args.skip_mcjq).to(device)
        if args.skip_mcjq:
            print("ABLATION: Skipping MCJQ — using static learned queries only")

    # ── Load checkpoints ─────────────────────────────────────────────
    if os.path.exists(args.heatmap_trained_path):
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
    else:
        print(f"WARN: heatmap checkpoint missing: {args.heatmap_trained_path}")

    if not args.skip_temporal:
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

    pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    if not args.skip_spatial:
        if not args.heatmap_path:
            raise ValueError("--heatmap_path is required unless --skip_spatial is set")
        if not args.skip_sjt and not args.spatial_transformer_path:
            raise ValueError("--spatial_transformer_path is required unless --skip_spatial or --skip_sjt is set")
        heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))
        if not args.skip_sjt:
            spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))

    for m in (net_heatmap, encoder, pose_decoder, heatmap_embedding, spatial_joint_transformer):
        m.eval()
    for p in net_heatmap.parameters():
        p.requires_grad = False

    models = (net_heatmap, encoder, heatmap_embedding, spatial_joint_transformer, pose_decoder)

    ablation_kwargs = dict(skip_temporal=args.skip_temporal, skip_spatial=getattr(args, 'skip_spatial', False),
                           skip_sjt=getattr(args, 'skip_sjt', False), hm_embed_dim=args.hm_embed_dim)

    # ── Benchmark (optional) ──────────────────────────────────────────
    if args.run_benchmark:
        sample_batch = next(iter(test_loader))
        sample_images = sample_batch['input_rgb'].to(device)
        print(f"\nBenchmark sample: {sample_images.shape}")

        report_params(models)
        benchmark(models, sample_images[:1], n_warmup=args.warmup, n_runs=args.bench_runs,
                  tag="(single-sequence, paper FPS)", **ablation_kwargs)
        if sample_images.shape[0] > 1:
            benchmark(models, sample_images, n_warmup=args.warmup, n_runs=args.bench_runs,
                      tag="(batch throughput)", **ablation_kwargs)

        if args.bench_only:
            return

    # ── Occlusion-bucketed analysis (R4 W2) ──────────────────────────
    # Uses heatmap peak confidence as a per-joint, per-frame visibility
    # proxy: low peak => the 2D detector is unsure => joint likely occluded
    # or out of view. We bucket per-joint errors by visibility and report
    # MPJPE in each bucket. Running this on the full model vs --skip_temporal
    # shows the motion stream helps most on low-visibility (occluded) joints.
    if args.occlusion_analysis:
        run_occlusion_analysis(models, test_loader, ablation_kwargs,
                               tag=args.occlusion_tag)
        return

    # ── Causal patch-occlusion probe (R4) ────────────────────────────
    # Actively occludes lower-body joints in a fraction of frames and sweeps
    # the fraction. Full vs --skip_temporal on the same masks shows temporal
    # recovery. Run both, matched --occ_seed, to build the comparison.
    if args.patch_occlusion:
        fracs = tuple(float(x) for x in args.occ_fracs.split(','))
        run_patch_occlusion(models, test_loader, ablation_kwargs,
                             tag=args.occlusion_tag, patch=args.occ_patch_size,
                             fracs=fracs, seed=args.occ_seed)
        return

    # ── Evaluation ───────────────────────────────────────────────────
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    total_mpjpe = 0.0
    total_pa_mpjpe = 0.0
    total_samples = 0
    if args.jointwise_results:
        joint_mpjpe_sum = torch.zeros(num_joints, device=device)
        joint_pa_mpjpe_sum = torch.zeros(num_joints, device=device)
    else:
        joint_mpjpe_sum = None
        joint_pa_mpjpe_sum = None

    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['input_rgb'].to(device)
            gt_egoposes = batch['gt_local_pose'].to(device)
            B, T = images.shape[:2]

            pose_logits = run_forward(*models, images, **ablation_kwargs)
            final = pose_logits.view(B, T, num_joints, 3)

            final_reshaped = final.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)

            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            S1_hat = batch_compute_similarity_transform_torch(final_reshaped, gt_reshaped)
            pa_mpjpe = mpjpe_loss_func(S1_hat, gt_reshaped)

            total_mpjpe += mpjpe_loss.item() * B * T
            total_pa_mpjpe += pa_mpjpe.item() * B * T
            total_samples += B * T
            if args.jointwise_results:
                joint_mpjpe = torch.linalg.norm(final_reshaped - gt_reshaped, dim=-1)
                joint_pa_mpjpe = torch.linalg.norm(S1_hat - gt_reshaped, dim=-1)
                joint_mpjpe_sum += joint_mpjpe.sum(dim=0)
                joint_pa_mpjpe_sum += joint_pa_mpjpe.sum(dim=0)

            if (i + 1) % args.log_step == 0:
                print(f'  [{i+1}/{len(test_loader)}] running MPJPE={total_mpjpe/total_samples:.4f}')

    avg_mpjpe = total_mpjpe / total_samples
    avg_pa_mpjpe = total_pa_mpjpe / total_samples
    if args.jointwise_results:
        joint_mpjpe_avg = (joint_mpjpe_sum / total_samples).detach().cpu().numpy()
        joint_pa_mpjpe_avg = (joint_pa_mpjpe_sum / total_samples).detach().cpu().numpy()
    else:
        joint_mpjpe_avg = None
        joint_pa_mpjpe_avg = None

    print(f'\n=== Final Results ===')
    print(f'MPJPE    : {avg_mpjpe:.4f}')
    print(f'PA-MPJPE : {avg_pa_mpjpe:.4f}')
    print(f'Samples  : {total_samples}')
    if args.jointwise_results:
        print('\n=== Jointwise Results ===')
        print(f"{'Joint':>2s}  {'Name':20s}  {'MPJPE':>10s}  {'PA-MPJPE':>10s}")
        for joint_idx, (mpjpe_value, pa_value) in enumerate(zip(joint_mpjpe_avg, joint_pa_mpjpe_avg)):
            joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
            print(f"{joint_idx:>2d}  {joint_name:20s}  {mpjpe_value:10.4f}  {pa_value:10.4f}")

    with open('test_results.txt', 'w') as f:
        f.write(f"MPJPE: {avg_mpjpe:.4f}\n")
        f.write(f"PA-MPJPE: {avg_pa_mpjpe:.4f}\n")
        f.write(f"Samples: {total_samples}\n")
        if args.jointwise_results:
            f.write("\nJointwise Results:\n")
            f.write("Joint,Name,MPJPE,PA-MPJPE\n")
            for joint_idx, (mpjpe_value, pa_value) in enumerate(zip(joint_mpjpe_avg, joint_pa_mpjpe_avg)):
                joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
                f.write(f"{joint_idx},{joint_name},{mpjpe_value:.4f},{pa_value:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    parser.add_argument('--model', type=str, default='egopw',
                        help='dataset/model option passed through to TestOptions')
    parser.add_argument('--encoder_path', type=str, default='')
    parser.add_argument('--decoder_path', type=str, required=True)
    parser.add_argument('--heatmap_trained_path', type=str, required=True)
    parser.add_argument('--heatmap_path', type=str, default='')
    parser.add_argument('--spatial_transformer_path', type=str, default='')
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
    parser.add_argument('--jointwise_results', action='store_true',
                        help='print and save per-joint MPJPE and PA-MPJPE')

    # Benchmarking
    parser.add_argument('--run_benchmark', action='store_true', help='run profiling (params, FLOPs, latency, FPS)')
    parser.add_argument('--warmup', type=int, default=10, help='warmup runs before latency timing')
    parser.add_argument('--bench_runs', type=int, default=50, help='timed runs for latency')
    parser.add_argument('--bench_only', action='store_true', help='run benchmark and skip full evaluation')
    parser.add_argument('--skip_temporal', action='store_true',
                        help='ablation: skip entire temporal stream (DINOv2 + ActionFormer)')
    parser.add_argument('--skip_spatial', action='store_true',
                        help='ablation: skip spatial stream (heatmaps/SJT)')
    parser.add_argument('--skip_sjt', action='store_true',
                        help='ablation: skip SpatialJointTransformer, use raw heatmap features')
    parser.add_argument('--concat_fusion', action='store_true',
                        help='ablation: concat + self-attention fusion instead of cross-attention/MCJQ')
    parser.add_argument('--skip_mcjq', action='store_true',
                        help='ablation: skip Motion-Conditioned Joint Queries, use static learned queries only')
    parser.add_argument('--freeze_actionformer', action='store_true',
                        help='ablation label only at test time — weights already baked into checkpoint')
    parser.add_argument('--random_actionformer', action='store_true',
                        help='ablation label only at test time — weights already baked into checkpoint')
    parser.add_argument('--mlp_decoder', action='store_true',
                        help='ablation: simple MLP decoder (concat + MLP, no transformer)')
    parser.add_argument('--temporal_baseline', type=str, default=None,
                        choices=['tcn', 'lstm'],
                        help='eval a from-scratch temporal baseline instead of ActionFormer')
    parser.add_argument('--causal', action='store_true',
                        help='causal/streaming temporal encoder (matches --temporal_baseline used at train)')
    parser.add_argument('--occlusion_analysis', action='store_true',
                        help='R4 W2: bucket per-joint MPJPE by heatmap-peak visibility, write CSV')
    parser.add_argument('--patch_occlusion', action='store_true',
                        help='R4: causal occluder-patch sweep over lower-body joints, write CSV')
    parser.add_argument('--occ_patch_size', type=int, default=40,
                        help='occluder patch size in pixels (in crop_size space)')
    parser.add_argument('--occ_fracs', type=str, default='0.0,0.25,0.5,0.75,1.0',
                        help='comma-separated fractions of frames to occlude for the patch sweep')
    parser.add_argument('--occ_seed', type=int, default=0,
                        help='RNG seed for occluded-frame selection (keep matched across full/skip_temporal runs)')
    parser.add_argument('--occlusion_tag', type=str, default='',
                        help='tag appended to the occlusion CSV filename (e.g. full, skip_temporal)')
    parser.add_argument('--dino_feature', type=str, default='cls',
                        choices=['cls', 'patch_mean'],
                        help='R2 W6 ablation: DINOv2 CLS token vs mean-pooled patch tokens')

    args = parser.parse_args()
    main(args)
