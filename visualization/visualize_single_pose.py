import argparse
import os
import sys
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.action_recognition import initialize_actionformer
from options.test_options import TestOptions
from utils.cross_attention_model import HeatmapToJointFeatures, SpatialJointTransformer, PoseDecoder
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.util import batch_compute_similarity_transform_torch

torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_joints = 15

print("Using device:", device)

SKELETON_CONNECTIONS = [
    (0, 1),   # Neck to Right_shoulder
    (0, 4),   # Neck to Left_shoulder
    (1, 2),   # Right_shoulder to Right_elbow
    (2, 3),   # Right_elbow to Right_wrist
    (4, 5),   # Left_shoulder to Left_elbow
    (5, 6),   # Left_elbow to Left_wrist
    (1, 7),   # Right_shoulder to Right_hip
    (4, 11),  # Left_shoulder to Left_hip
    (7, 8),   # Right_hip to Right_knee
    (8, 9),   # Right_knee to Right_ankle
    (9, 10),  # Right_ankle to Right_foot
    (11, 12), # Left_hip to Left_knee
    (12, 13), # Left_knee to Left_ankle
    (13, 14), # Left_ankle to Left_foot
    (7, 11),  # Right_hip to Left_hip
]

JOINT_NAMES = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist",
    "Left_shoulder", "Left_elbow", "Left_wrist", "Right_hip",
    "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
    "Left_knee", "Left_ankle", "Left_foot"
]

def plot_3d_pose(poses, title="3D Pose", connections=None, ax=None, color='darkblue', alpha=0.7, line_color=None):
    """Plot 3D pose with skeleton connections"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    if line_color is None:
        line_color = color

    ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c=color, s=160, alpha=alpha)

    if connections is not None:
        for start_idx, end_idx in connections:
            if start_idx < len(poses) and end_idx < len(poses):
                start_point = poses[start_idx]
                end_point = poses[end_idx]
                ax.plot([start_point[0], end_point[0]],
                       [start_point[1], end_point[1]],
                       [start_point[2], end_point[2]],
                       color=line_color, linewidth=4, alpha=alpha)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    max_range = np.array([poses[:,0].max()-poses[:,0].min(),
                         poses[:,1].max()-poses[:,1].min(),
                         poses[:,2].max()-poses[:,2].min()]).max() / 2.0
    mid_x = (poses[:,0].max()+poses[:,0].min()) * 0.5
    mid_y = (poses[:,1].max()+poses[:,1].min()) * 0.5
    mid_z = (poses[:,2].max()+poses[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    requires_gt = args.dataset == 'sceneego'
    if requires_gt and args.gt_pickle_path is None:
        raise ValueError("--gt_pickle_path is required for SceneEgo visualization")
    if args.gt_pickle_path is not None and not os.path.exists(args.gt_pickle_path):
        gt_dir = os.path.dirname(args.gt_pickle_path)
        alternatives = [
            os.path.join(gt_dir, name)
            for name in ("annotation.pkl", "local_pose_gt.pkl", "pose_gt.pkl")
            if os.path.exists(os.path.join(gt_dir, name))
        ]
        suggestion = f" Try one of: {', '.join(alternatives)}" if alternatives else ""
        raise FileNotFoundError(f"Ground-truth pickle not found: {args.gt_pickle_path}.{suggestion}")

    print(f"Loading image from {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    # Duplicate single image T times to form a sequence — matches training setup
    images = image_tensor.unsqueeze(1).repeat(1, args.seq_length, 1, 1, 1)  # [1, T, 3, H, W]

    # ── Initialize ActionFormer (same as test.py) ─────────────────────────
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    for param in actionformer_feature_extractor.parameters():
        param.requires_grad = False

    opt = TestOptions().parse()

    # ── Build models (matching train.py exactly) ──────────────────────────
    net_heatmap = HeatMap_Network(opt, model_name=args.heatmap_backbone).to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=args.hm_embed_dim, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor, skip_temporal=False).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)

    # ── Load checkpoints ──────────────────────────────────────────────────
    print(f"Loading heatmap network from {args.heatmap_trained_path}")
    net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))

    print(f"Loading encoder from {args.encoder_path}")
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

    print(f"Loading pose decoder from {args.decoder_path}")
    pose_decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    print(f"Loading heatmap embedding from {args.heatmap_path}")
    heatmap_embedding.load_state_dict(torch.load(args.heatmap_path, map_location=device))

    print(f"Loading spatial transformer from {args.spatial_transformer_path}")
    spatial_joint_transformer.load_state_dict(torch.load(args.spatial_transformer_path, map_location=device))

    # ── Set all models to eval ────────────────────────────────────────────
    for m in (net_heatmap, encoder, pose_decoder, heatmap_embedding, spatial_joint_transformer):
        m.eval()
    for param in net_heatmap.parameters():
        param.requires_grad = False

    print("All models loaded. Running inference...")

    with torch.no_grad():
        B, T, _, H_img, W_img = images.shape
        H_hm, W_hm = 64, 64

        # ── Spatial stream (mirrors train.py lines 341-351) ───────────────
        all_images_flat = images.view(-1, 3, H_img, W_img)       # [B*T, 3, H, W]
        all_heatmaps = net_heatmap(all_images_flat)               # [B*T, J, H_hm, W_hm]
        all_heatmaps = torch.sigmoid(all_heatmaps)
        heatmaps = all_heatmaps.view(B, T, 15, H_hm, W_hm)       # [B, T, 15, 64, 64]

        heatmap_features = heatmap_embedding(heatmaps)            # (B, T, J, 128)
        spatial_joint_features = spatial_joint_transformer(heatmap_features)  # (B, T, J, 128)

        # ── Temporal stream (mirrors train.py lines 331-333) ─────────────
        motion_features = encoder(images)                         # (B, T, 384)
        motion_features = motion_features.float()

        # ── Decode (PoseDecoder already returns (B,T,J,3)) ───────────────
        pose_logits = pose_decoder(spatial_joint_features, motion_features)  # (B, T, J, 3)
        final = pose_logits.view(B, T, num_joints, 3)

        # Use middle frame (most representative for a duplicated sequence)
        frame_idx = T // 2
        pred_pose_tensor = final[0, frame_idx:frame_idx+1]        # [1, 15, 3]

        print(f"Predicted pose shape: {pred_pose_tensor.shape}, frame {frame_idx}/{T}")

        # ── Load ground truth when provided ───────────────────────────────
        gt_pose = None
        if args.gt_pickle_path is not None and os.path.exists(args.gt_pickle_path):
            image_filename = os.path.basename(args.image_path)
            print(f"Loading pickle: {args.gt_pickle_path}")
            print(f"Searching for: {image_filename}")
            with open(args.gt_pickle_path, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {len(data)} entries")
            for item in data:
                if item.get('image_name') == image_filename:
                    if args.dataset == 'sceneego':
                        gt_pose = item.get('ego_pose_gt')
                    else:
                        gt_pose = item.get('optimized_local_pose')
                    print(f"Found ground truth for {image_filename}")
                    break
            if gt_pose is None:
                if requires_gt:
                    raise ValueError(
                        f"Ground truth for {image_filename} was not found in {args.gt_pickle_path}. "
                        "For SceneEgo test clips, use annotation.pkl because local_pose_gt.pkl does not store image_name."
                    )
                print(f"Ground truth for {image_filename} was not found in {args.gt_pickle_path}; using raw prediction.")

        # ── Procrustes alignment (always applied when GT available) ───────
        pred_pose_np = pred_pose_tensor[0].cpu().numpy()  # raw prediction
        if gt_pose is not None:
            gt_pose_tensor = torch.from_numpy(gt_pose).unsqueeze(0).float().to(device)
            aligned_pose_tensor = batch_compute_similarity_transform_torch(pred_pose_tensor, gt_pose_tensor)
            aligned_pose = aligned_pose_tensor[0].cpu().numpy()

            mpjpe_before = torch.sqrt(((pred_pose_tensor - gt_pose_tensor) ** 2).sum(dim=-1)).mean().item()
            mpjpe_after  = torch.sqrt(((aligned_pose_tensor - gt_pose_tensor) ** 2).sum(dim=-1)).mean().item()
            print(f"MPJPE          : {mpjpe_before:.2f} mm")
            print(f"PA-MPJPE       : {mpjpe_after:.2f} mm")
        else:
            aligned_pose = pred_pose_np
            mpjpe_before = mpjpe_after = None
            print("No GT found — Procrustes skipped, displaying raw prediction")

        def _make_fig():
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            return fig, ax

        def _fit_axes(ax, *pose_arrays):
            all_p = np.concatenate(pose_arrays, axis=0)
            r = np.array([all_p[:,0].max()-all_p[:,0].min(),
                          all_p[:,1].max()-all_p[:,1].min(),
                          all_p[:,2].max()-all_p[:,2].min()]).max() / 2.0
            cx, cy, cz = all_p[:,0].mean(), all_p[:,1].mean(), all_p[:,2].mean()
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_zlim(cz - r, cz + r)

        if args.dataset == 'sceneego' and gt_pose is not None:
            # Ground truth (red) superimposed with Procrustes-aligned prediction.
            fig, ax = _make_fig()
            plot_3d_pose(gt_pose, "", SKELETON_CONNECTIONS, ax,
                        color='red', alpha=0.85, line_color='darkred')
            plot_3d_pose(aligned_pose, "", SKELETON_CONNECTIONS, ax,
                        color='darkblue', alpha=0.8, line_color='#32CD32')
            _fit_axes(ax, gt_pose, aligned_pose)
            ax.legend(handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Ground Truth', alpha=0.8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
                       markersize=10, label=f'Procrustes-Aligned (PA-MPJPE={mpjpe_after:.1f}mm)', alpha=0.8),
            ], loc='upper right')
            plt.tight_layout()
            plt.show()

        else:
            # EgoPW: Procrustes-aligned prediction only (GT used for alignment, not shown)
            fig, ax = _make_fig()
            plot_3d_pose(aligned_pose, "", SKELETON_CONNECTIONS, ax,
                        color='darkblue', alpha=0.8, line_color='#32CD32')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str,
                        default='/data/My_Backup/Dataset/SceneEgo_test/diogo1/imgs/img_002096.jpg')
    parser.add_argument('--gt_pickle_path', type=str, default=None,
                        help='Path to pickle file with ground truth poses (required for sceneego)')

    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml')
    parser.add_argument('--encoder_path', type=str, required=True)
    parser.add_argument('--decoder_path', type=str, required=True)
    parser.add_argument('--heatmap_trained_path', type=str, required=True)
    parser.add_argument('--heatmap_path', type=str, required=True)
    parser.add_argument('--spatial_transformer_path', type=str, required=True)

    parser.add_argument('--heatmap_backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--hm_embed_dim', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--seq_length', type=int, default=64,
                        help='sequence length — single image is duplicated T times')
    parser.add_argument('--dataset', type=str, default='sceneego',
                        choices=['egopw', 'sceneego'],
                        help='egopw: prediction only; sceneego: show GT + Procrustes-aligned prediction')

    args = parser.parse_args()
    main(args)
