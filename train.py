# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
from sched import scheduler

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils.data_loader import get_loader
from utils.build_annotation import Annotation
from action_recognition import ActionFormerFeatureExtractor
from action_recognition import initialize_actionformer
from utils.model import EncoderCNN, DecoderRNN, TemporalGCN
from torch.nn.utils.rnn import pack_padded_sequence
from utils.build_annotation import Annotation
from utils.build_validation_annotation import ValidationAnnotation
from torchvision import transforms
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OS:", os.cpu_count())
print("Working On:",device)
# Number of joints and coordinates per joint
num_joints = 25
coords_per_joint = 3

# Define bidirectional connections for the central body part
bidirectional_connections = [
    (0, 1),  # SpineBase <-> SpineMid
    (1, 20), # SpineMid <-> SpineShoulder
    (20, 2), # SpineShoulder <-> Neck
    (2, 3),  # Neck <-> Head
    (0, 12), # SpineBase <-> HipLeft
    (0, 16)  # SpineBase <-> HipRight
]

unidirectional_connections = [
    # Left arm chain
    (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),
    # Right arm chain
    (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),
    # Left leg chain
    (12, 13), (13, 14), (14, 15),
    # Right leg chain
    (16, 17), (17, 18), (18, 19)
]

edge_index = [[], []]

for joint_a, joint_b in bidirectional_connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)
        edge_index[1].append(joint_b * coords_per_joint + j)
        edge_index[0].append(joint_b * coords_per_joint + j)
        edge_index[1].append(joint_a * coords_per_joint + j)

# Add unidirectional edges (one direction for each pair)
for joint_a, joint_b in unidirectional_connections:
    for j in range(coords_per_joint):
        edge_index[0].append(joint_a * coords_per_joint + j)
        edge_index[1].append(joint_b * coords_per_joint + j)

# Optionally, add self-loops for each joint
for joint in range(num_joints):
    for j in range(coords_per_joint):
        edge_index[0].append(joint * coords_per_joint + j)
        edge_index[1].append(joint * coords_per_joint + j)

edge_index = torch.tensor(edge_index, dtype=torch.long)

def mean_per_joint_position_error(predicted, target):
    print(f"Predicted shape: {predicted.shape}")
    print(f"Target shape: {target.shape}")

    predicted = predicted.view(predicted.size(0), 25, 3)
    target = target.view(target.size(0), 25, 3)
    error = torch.norm(predicted - target, dim=-1)
    mpjpe = torch.mean(error)
    return mpjpe.item()

def evaluate(encoder, decoder, data_loader, criterion):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_mpjpe = 0
    with torch.no_grad():
        for images, gt_egoposes, homography, poses2, lengths in data_loader:
            images = images.to(device)
            gt_egoposes = gt_egoposes.to(device)

            features = encoder(images)
            outputs = decoder(features, lengths, homography, poses2)

            targets = pack_padded_sequence(gt_egoposes, lengths, batch_first=True)[0]
            mpjpe = mean_per_joint_position_error(outputs, targets)
            total_mpjpe += mpjpe
        avg_mpjpe = total_mpjpe / len(data_loader)
        print(f'Average MPJPE: {avg_mpjpe:.4f}')

    return avg_mpjpe

def main(args):
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

    with open(args.annotation_path, 'rb') as f:
        annotation = pickle.load(f)

    with open(args.validation_annotation_path, 'rb') as f:
        validation_annotation = pickle.load(f)
    # build data loader
    data_loader = get_loader(annotation, args.image_dir, args.h_dir, args.openpose_dir, transform,
                             args.batch_size, shuffle=True, num_workers=args.num_workers, seq_length=args.seq_length)

    val_loader = get_loader(validation_annotation, args.image_dir, args.h_dir, args.openpose_dir, transform,
                             args.batch_size, shuffle=False, num_workers=args.num_workers, seq_length=args.seq_length)

    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    print("Total Validation dataset", len(val_loader.dataset))
    encoder = EncoderCNN(args.embed_size,actionformer_feature_extractor).to(device)
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

    # loss and optimizer
    criterion = nn.MSELoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    total_step = len(data_loader)
    best_val_loss = float('inf')
    model_dir = os.path.abspath('./utils/trained_ckpt_actionformer')
    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
        for i, (images, gt_egoposes, homography, poses2, lengths) in enumerate(data_loader):
            print("Printing iteration number:", i)
            images = images.to(device)
            print("Image shape", images.shape)
            gt_egoposes = gt_egoposes.to(device)
            targets = pack_padded_sequence(gt_egoposes, lengths, batch_first=True)[0]
            features = encoder(images)
            print("Encoded feature shape", features.shape)
            outputs = decoder(features, lengths, homography, poses2)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"NaN detected! Epoch: {epoch}, Iteration: {i}")
                break
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_value)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_value)
            optimizer.step()
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

    # File to store evaluation results
    eval_results_file = os.path.join('best_model_evaluation.txt')
    with open(eval_results_file, 'w') as f:
        f.write("Epoch\tValidation Loss\n")
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
        encoder.train()
        decoder.train()
        for i, (images, gt_egoposes, homography, poses2, lengths) in enumerate(data_loader):
            print("Printing iteration number:", i)
            images = images.to(device)
            gt_egoposes = gt_egoposes.to(device)
            # Squeeze the targets to make them 1D
            targets = pack_padded_sequence(gt_egoposes, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, lengths, homography, poses2)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"NaN detected! Epoch: {epoch}, Iteration: {i}")
                break
            # Backward and optimize with mixed precision
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_value)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_value)
            optimizer.step()
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

        val_mpjpe = evaluate(encoder, decoder, val_loader, criterion)
        scheduler.step(val_mpjpe)
        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        print(f'Epoch [{epoch + 1}] Validation Loss: {val_mpjpe:.4f}')

        # Save the model only if validation loss improves
        if val_mpjpe < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_mpjpe:.4f}, saving model...")
            best_val_loss = val_mpjpe
            torch.save(decoder.state_dict(), os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

            # Log the best validation loss
            with open(eval_results_file, 'a') as f:
                f.write(f'{epoch + 1}\t{val_mpjpe:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/anet_tsp.yaml', help='path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')
    parser.add_argument('--validation_annotation_path', type=str, required=True, help='path for validation annotation wrapper')

    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')

    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=5, help='step size for saving trained models')

    args = parser.parse_args()
    print(args)
    main(args)
