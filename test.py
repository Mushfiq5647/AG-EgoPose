import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils.data_loader import get_loader
from utils.build_vocab import Vocabulary
from utils.build_test_annotation import TestAnnotation
from utils.model import EncoderCNN, DecoderRNN
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_per_joint_position_error(predicted, target):
    # Reshape the data from (batch_size, sequence_length, 75) to (batch_size, sequence_length, 25, 3)
    print(f"Predicted shape: {predicted.shape}")
    print(f"Target shape: {target.shape}")

    predicted = predicted.view(predicted.size(0), 19, 3)
    target = target.view(target.size(0), 19, 3)

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

    with open(args.test_annotation_path, 'rb') as f:
        annotation = pickle.load(f)

    # Build data loader
    data_loader = get_loader(annotation, args.image_dir, args.h_dir, args.openpose_dir, transform,
                             args.batch_size, shuffle=False, num_workers=args.num_workers, seq_length=args.seq_length)

    print("Batch size", len(data_loader.dataset))
    # Load models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         args.seq_length,
                         args.num_layers).to(device)

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

            # Calculate MPJPE
            mpjpe = mean_per_joint_position_error(outputs, targets)
            total_mpjpe += mpjpe

            if (i + 1) % args.log_step == 0:
                print(f'Batch [{i + 1}/{len(data_loader)}], MPJPE: {mpjpe:.4f}')

        avg_mpjpe = total_mpjpe / len(data_loader)
        print(f'Average MPJPE: {avg_mpjpe:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument('--encoder_path', type=str, required=True, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
    parser.add_argument('--test_annotation_path', type=str, required=True, help='path for annotation wrapper')

    # Directories
    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_cmu/cmu', help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_cmu/cmu', help='directory for homography')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_cmu/cmu', help='directory for OpenPose JSON files')

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
