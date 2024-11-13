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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("OS:", os.cpu_count())
print("Working On:",device)
num_joints = 15
coords_per_joint = 3

# Define a joint hierarchy based on an assumed structure
connections = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
         (11, 12), (12, 13), (13, 14), (7, 11)]
# Initialize lists for edge indices
edge_index = [[], []]

# Add unidirectional edges (one direction for each pair)
for joint_a, joint_b in connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)
        edge_index[1].append(joint_b * coords_per_joint + j)

# Optionally, add self-loops for each joint
for joint in range(num_joints):
    for j in range(coords_per_joint):
        edge_index[0].append(joint * coords_per_joint + j)
        edge_index[1].append(joint * coords_per_joint + j)

print("Edge index:", edge_index)
# Convert to tensor
edge_index = torch.tensor(edge_index, dtype=torch.long)

def mean_per_joint_position_error(predicted, target):
    # Reshape the data from (batch_size, sequence_length, 75) to (batch_size, sequence_length, 25, 3)
    print(f"Predicted shape: {predicted.shape}")
    print(f"Target shape: {target.shape}")

    predicted = predicted.view(predicted.size(0), 15, 3)
    target = target.view(target.size(0), 15, 3)

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
    total_samples = 0
    with torch.no_grad():
        for i, (images, gt_egoposes, lengths) in enumerate(data_loader):
            # Get batch size for this iteration
            current_batch_size = images.size(0)  # Will be 16 for full batches, 12 for last batch
            images = images.to(device)
            gt_egoposes = gt_egoposes.to(device)
            targets = pack_padded_sequence(gt_egoposes, lengths, batch_first=True)[0]
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, lengths)
            print("Outputs:", outputs.shape)
            print("Targets:", targets.shape)
            outputs = outputs.view(-1, 57)

            print(f"Batch {i + 1}:")
            print(f"Current batch size: {current_batch_size}")
            print(f"Outputs shape: {outputs.shape}")  # Should be (current_total_frames, 45)
            print(f"Targets shape: {targets.shape}")

            # Calculate MPJPE for this batch
            mpjpe = mean_per_joint_position_error(outputs, targets)

            # Weight the MPJPE by the actual number of sequences in this batch
            total_mpjpe += mpjpe * current_batch_size
            total_samples += current_batch_size

            if (i + 1) % args.log_step == 0:
                print(f'Batch [{i + 1}/{len(data_loader)}], '
                      f'Batch Size: {current_batch_size}, '
                      f'MPJPE: {mpjpe:.4f}')

    # Calculate weighted average MPJPE
    avg_mpjpe = total_mpjpe / total_samples  # Dividing by actual number of sequences (540)
    print(f'Total sequences processed: {total_samples}')  # Should print 540
    print(f'Average MPJPE: {avg_mpjpe:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument('--config_path', type=str, default='actionformer/config/anet_tsp.yaml', help='path to the config file')
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--test_annotation_path', type=str, required=True, help='path for annotation wrapper')

    # Directories
    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_cmu/cmu',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_cmu/cmu',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_cmu/cmu',
                        help='directory for resized images')

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
