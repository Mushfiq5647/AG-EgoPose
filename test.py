import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils.data_loader import get_loader
from utils.build_vocab import Vocabulary
from utils.build_test_annotation import TestAnnotation
from utils.model import EncoderCNN, DecoderRNN,TemporalGCN
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from action_recognition import initialize_actionformer
from copy import deepcopy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_joints = 25
coords_per_joint = 3

# Define connections between joints (based on the skeleton structure)
connections = [
    (0, 12), (0, 16), (0, 1),  # SpineBase to HipLeft, HipRight, SpineMid
    (1, 20),  # SpineMid to SpineShoulder
    (20, 2),  # SpineShoulder to Neck
    (2, 3),  # Neck to Head
    (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),  # Left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),  # Right arm
    (12, 13), (13, 14), (14, 15),  # Left leg
    (16, 17), (17, 18), (18, 19),  # Right leg
]

# Create the edge_index
edge_index = [[], []]

# For each connection, append the corresponding indices for all 3D coordinates (x, y, z)
for joint_a, joint_b in connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)  # Source joint's coordinate index
        edge_index[1].append(joint_b * coords_per_joint + j)  # Target joint's coordinate index

# Convert to tensor
edge_index = torch.tensor(edge_index, dtype=torch.long)

def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n



    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t
def align_skeleton(estimated_seq, gt_seq, skeleton_model=None, scale=True):
    estimated_seq = deepcopy(np.asarray(estimated_seq))
    gt_seq = deepcopy(np.asarray(gt_seq))
    if skeleton_model is not None:
        for i in range(len(estimated_seq)):
            estimated_seq[i] = skeleton_model.skeleton_resize_single(
                estimated_seq[i],
                bone_length_file='utils/fisheye/mean3D.mat')
        for i in range(len(gt_seq)):
            gt_seq[i] = skeleton_model.skeleton_resize_single(
                gt_seq[i],
                bone_length_file='utils/fisheye/mean3D.mat')

    aligned_pose_list = np.zeros_like(estimated_seq)
    for s in range(estimated_seq.shape[0]):
        pose_p = estimated_seq[s]
        pose_gt_bs = gt_seq[s]
        if scale is False:
            # if scale is False, firstly align the center of each pose
            pose_p_center = np.mean(pose_p, axis=0)
            pose_gt_center = np.mean(pose_gt_bs, axis=0)
            pose_p -= pose_p_center
            pose_gt_bs -= pose_gt_center

        c, R, t = umeyama(pose_p, pose_gt_bs)
        if scale is True:
            pose_p = pose_p.dot(R) * c + t
        else:
            pose_p = pose_p.dot(R) + t
        aligned_pose_list[s] = pose_p

    return aligned_pose_list, gt_seq

def mean_per_joint_position_error(predicted, target):
    # Reshape the data from (batch_size, sequence_length, 75) to (batch_size, sequence_length, 25, 3)
    print(f"Predicted shape: {predicted.shape}")
    print(f"Target shape: {target.shape}")

    predicted = predicted.view(predicted.size(0), 25, 3)
    target = target.view(target.size(0), 25, 3)

    # Compute the Euclidean distance (L2 norm) between predicted and target for each joint
    error = torch.norm(predicted - target, dim=-1)  # L2 norm along the last dimension (x, y, z)

    # Compute the mean over all joints, frames, and batches
    mpjpe = torch.mean(error)  # Average over all joints and frames
    return mpjpe.item()


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)

    with open(args.test_annotation_path, 'rb') as f:
        annotation = pickle.load(f)

    data_loader = get_loader(annotation, args.image_dir, args.h_dir, args.openpose_dir, transform,
                             args.batch_size, shuffle=False, num_workers=args.num_workers, seq_length=args.seq_length)

    print("Batch size", len(data_loader.dataset))
    # Load models
    encoder = EncoderCNN(args.embed_size, actionformer_feature_extractor).to(device)

    temporal_gcn = TemporalGCN(
        args.hidden_size,
        args.seq_length,
        edge_index,
        output_dim=75,
        kernel_size=7).to(device)

    decoder = DecoderRNN(args.embed_size,
                         args.hidden_size,
                         args.seq_length,
                         args.num_layers,
                         temporal_gcn).to(device)

    # Load trained models
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    encoder.eval()
    decoder.eval()

    total_mpjpe = 0
    with torch.no_grad():
        for i, (images, gt_egoposes, homography, poses2, lengths) in enumerate(data_loader):
            images = images.to(device)
            gt_egoposes = gt_egoposes.to(device)
            targets = pack_padded_sequence(gt_egoposes, lengths, batch_first=True)[0]
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, lengths, homography, poses2)
            print("Outputs:", outputs.shape)
            print("Targets:", targets.shape)
            outputs = outputs.view(-1, 75)

            # Calculate MPJPE
            mpjpe = mean_per_joint_position_error(outputs, targets)
            aligned_outputs, groundtruth = align_skeleton(outputs, targets, None)
            pa_mpjpe = mean_per_joint_position_error(aligned_outputs, groundtruth)
            total_mpjpe += mpjpe
            total_pa_pjpe = pa_mpjpe
            if (i + 1) % args.log_step == 0:
                print(f'Batch [{i + 1}/{len(data_loader)}], MPJPE: {mpjpe:.4f}')

        avg_mpjpe = total_mpjpe / len(data_loader)
        avg_pa_mpjpe = total_pa_pjpe / len(data_loader)
        print(f'Average MPJPE: {avg_mpjpe:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument('--config_path', type=str, default='actionformer/config/anet_tsp.yaml', help='path to the config file')
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--test_annotation_path', type=str, required=True, help='path for annotation wrapper')

    # Directories
    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for homography')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect', help='directory for OpenPose JSON files')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')

    # Other settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')

    args = parser.parse_args()
    main(args)
