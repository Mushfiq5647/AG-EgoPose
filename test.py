# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast

from action_recognition import initialize_actionformer
from options.test_options import TestOptions
from utils.data_loader import dataloader_full
from utils.cross_attention_model import HeatmapToJointFeatures
from utils.cross_attention_model import SpatialJointTransformer
from utils.cross_attention_model import PoseDecoder
from utils.loss import LossFuncMPJPE
from heatmaps.network_heatmap import HeatMap_Network
from utils.model import FeatureEncoder
from utils.util import batch_compute_similarity_transform_torch
from utils.pose_evaluation import calculate_ba_mpjpe
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

# Number of joints and coordinates per joint
num_joints = 15
# coords_per_joint = 3
# # Device configuration
#
# # Define connections between joints (based on the skeleton structure)
# connections = [
#     (0, 12), (0, 16), (0, 1),  # SpineBase to HipLeft, HipRight, SpineMid
#     (1, 20),  # SpineMid to SpineShoulder
#     (20, 2),  # SpineShoulder to Neck
#     (2, 3),  # Neck to Head
#     (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),  # Left arm
#     (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),  # Right arm
#     (12, 13), (13, 14), (14, 15),  # Left leg
#     (16, 17), (17, 18), (18, 19),  # Right leg
# ]
#
# # Create the edge_index
# edge_index = [[], []]
#
# # For each connection, append the corresponding indices for all 3D coordinates (x, y, z)
# for joint_a, joint_b in connections:
#     for j in range(coords_per_joint):
#         edge_index[0].append(joint_a * coords_per_joint + j)  # Source joint's coordinate index
#         edge_index[1].append(joint_b * coords_per_joint + j)  # Target joint's coordinate index
#
# # Convert to tensor
# edge_index = torch.tensor(edge_index, dtype=torch.long)

def egoglobal_to_egopw(vec):
    """
    vec: (..., 3) [Xg, Yg, Zg]
    EgoGlobal -> EgoPW axis conversion
    """
    # Step 1: permute axes (Xg,Yg,Zg) -> (Zg,Xg,Yg)
    v = vec[..., [2, 0, 1]]
    # Step 2: negate Y and Z
    v[..., 1] = -v[..., 1]
    v[..., 2] = -v[..., 2]
    return v



def main(args):
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
    # Initialize models (same as train.py)
    net_heatmap = HeatMap_Network(opt, model_name='resnet18').to(device)
    heatmap_embedding = HeatmapToJointFeatures(heatmap_size=64, feature_dim=128, method='conv_pool').to(device)
    encoder = FeatureEncoder(actionformer_feature_extractor).to(device)
    spatial_joint_transformer = SpatialJointTransformer(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    # pose_decoder = PoseDecoder(motion_dim=384, joint_dim=128, out_dim=3).to(device)
    pose_decoder = PoseDecoder(args.hm_embed_dim, num_heads=4, num_layers=3).to(device)
    # Load pre-trained 2D heatmap network
    if os.path.exists(args.heatmap_trained_path):
        print(f"Loading pre-trained 2D heatmap network from {args.heatmap_trained_path}")
        net_heatmap.load_state_dict(torch.load(args.heatmap_trained_path, map_location=device))
        net_heatmap.eval()
        # Freeze the heatmap network
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

    # Initialize loss functions
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    
    # Initialize evaluation metrics
    total_mpjpe = 0
    total_procrustes_error = 0.0
    total_ba_mpjpe_error = 0.0
    total_samples = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, (batch) in enumerate(test_loader):
            print("Printing iteration number:", i)
            # Instead of: for i, (images, homography, gt_egoposes, lengths) in enumerate(data_loader)
            images = batch['input_rgb'].to(device)  # Tensor
            B, T, _, H_img, W_img = images.shape
            H_hm, W_hm = 64, 64
            gt_egoposes = batch['gt_local_pose'].to(device)
            # Process ALL images at once (more efficient)
            with torch.no_grad():
                all_images_flat = images.view(-1, 3, H_img, W_img)  # [B*T, 3, H, W]
                print("Image shape", all_images_flat.shape)
                all_heatmaps = net_heatmap(all_images_flat)  # [B*T, J, H_hm, W_hm]
                all_heatmaps = torch.sigmoid(all_heatmaps)  # Convert logits to 0-1 range for BCE with logits model
                heatmaps = all_heatmaps.view(B, T, 15, H_hm, W_hm)  # [B,
                # T, 15, H_hm, W_hm]
            motion_features = encoder(images)
            heatmap_features = heatmap_embedding(heatmaps)
            spatial_joint_features = spatial_joint_transformer(heatmap_features)
            print("Passed spatial_joint_features")
            pose_logits = pose_decoder(heatmap_features, motion_features)
            print("Passed decoder")

            # Reshape to pose format
            final = pose_logits.view(B, T, num_joints, 3)

            # Reshape for loss computation
            final_reshaped = final.reshape(B * T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B * T, num_joints, 3)
            sample_idx = 0


            print("Ground truth joints (first 5):")
            print(gt_reshaped[sample_idx, :14])
            
            # Compute MPJPE loss
            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            
            print(f"MPJPE: {mpjpe_loss:.4f}")
            
            print("GT Poses", gt_egoposes.shape)
            print("Gt Poses min/max:", gt_egoposes.min().item(), gt_egoposes.max().item())
            print("Outputs Final", final.shape)

            # Calculate Procrustes alignment for PA-MPJPE
            S1_hat = batch_compute_similarity_transform_torch(final_reshaped, gt_reshaped)
            print("Predicted joints (first 5):")
            print(S1_hat[sample_idx, :14])  # first 5 joints
            
            # Calculate PA-MPJPE (Procrustes Aligned MPJPE)
            pa_mpjpe = mpjpe_loss_func(S1_hat, gt_reshaped)
            
            # Calculate BA-MPJPE using new Umeyama approach
            # final_np = final_reshaped.detach().cpu().numpy()
            # gt_np = gt_reshaped.detach().cpu().numpy()
            # ba_mpjpe = calculate_ba_mpjpe(final_np, gt_np, skeleton_model=None)
            
            print(f'Batch [{i + 1}/{len(test_loader)}], MPJPE: {mpjpe_loss:.4f}, PA-MPJPE: {pa_mpjpe:.4f}')
            
            # Accumulate errors
            total_mpjpe += mpjpe_loss.item() * B * T
            total_procrustes_error += pa_mpjpe.item() * B * T
            # total_ba_mpjpe_error += ba_mpjpe * B * T
            total_samples += B * T

        # Calculate final averages
        avg_mpjpe = total_mpjpe / total_samples
        avg_pa_mpjpe = total_procrustes_error / total_samples
        # avg_ba_mpjpe = total_ba_mpjpe_error / total_samples
        
        print(f'\n=== Final Evaluation Results ===')
        print(f'Average MPJPE: {avg_mpjpe:.4f}')
        print(f'Average PA-MPJPE: {avg_pa_mpjpe:.4f}')
        # print(f'Average BA-MPJPE: {avg_ba_mpjpe:.4f}')
        print(f'Total samples evaluated: {total_samples}')
        
        # Save results to file
        results_file = 'test_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Test Results Summary\n")
            f.write(f"===================\n")
            f.write(f"Average MPJPE: {avg_mpjpe:.4f}\n")
            f.write(f"Average PA-MPJPE: {avg_pa_mpjpe:.4f}\n")
            # f.write(f"Average BA-MPJPE: {avg_ba_mpjpe:.4f}\n")
            f.write(f"Total samples evaluated: {total_samples}\n")
        
        print(f"Results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print(sys.argv)
    # Model and data paths
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml',
                        help='path to the config file')
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
    parser.add_argument('--heatmap_path', type=str, required=True, help='path for trained heatmap embedding')
    parser.add_argument('--spatial_transformer_path', type=str, required=True, help='path for trained spatial transformer')
    # parser.add_argument('--test_annotation_path', type=str, required=True, help='path for annotation wrapper')

    # Directories
    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for homography')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for OpenPose JSON files')

    # Model parameters
    parser.add_argument('--embed_feature_dim', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hm_embed_dim', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--seq_length', type=int, default=32, help='length of the pose/video sequences')

    # Other settings
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=256, help='size for randomly cropping images')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')

    args = parser.parse_args()
    main(args)
