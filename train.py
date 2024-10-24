# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.build_annotation import Annotation
from utils.build_validation_annotation import ValidationAnnotation
from torchvision import transforms

from utils.data_loader import get_loader
from utils.model import EncoderCNN, DecoderRNN

torch.set_printoptions(threshold=torch.inf)
# Device configuration
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OS:", os.cpu_count())
print("Working On:", device)
import os

# Evaluation function

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
    # create model directory
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
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size,
                         args.hidden_size,
                         args.seq_length,
                         args.num_layers).to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float('inf')
    model_dir = os.path.abspath('./utils/trained_ckpt_you2me')

    # File to store evaluation results
    eval_results_file = os.path.join('best_model_evaluation.txt')
    with open(eval_results_file, 'w') as f:
        f.write("Epoch\tMPJPE\n")
    total_step = len(data_loader)
    # print ("total iter", total_step)
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
        print(f'Epoch [{epoch + 1}] Validation Loss: {val_mpjpe:.4f}')
        scheduler.step(val_mpjpe)
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']  # Access the learning rate of the first parameter group
        print(f'Current Learning Rate: {current_lr:.6f}')

        # Save the model only if validation loss improves
        if val_mpjpe < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_mpjpe:.4f}, saving model...")
            best_val_loss = val_mpjpe
            torch.save(decoder.state_dict(), os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

            # Log the best validation loss
            with open(eval_results_file, 'a') as f:
                f.write(f'{epoch + 1}\t\t{val_mpjpe:.4f}\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')
    parser.add_argument('--validation_annotation_path', type=str, required=True, help='path for validation annotation wrapper')

    parser.add_argument('--upp', action='store_true', help='set flag if training upper body model')
    parser.add_argument('--low', action='store_true', help='set flag if training lower body model')

    parser.add_argument('--image_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')

    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=0.0003)
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
































