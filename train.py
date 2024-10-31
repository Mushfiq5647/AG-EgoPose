# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.build_annotation import Annotation
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

    # build data loader
    data_loader = get_loader(annotation, args.image_dir, args.h_dir, args.openpose_dir, transform,
                             args.batch_size, shuffle=True, num_workers=args.num_workers, seq_length=args.seq_length)


    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size,
                         args.hidden_size,
                         args.seq_length,
                         args.num_layers).to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    model_dir = os.path.abspath('./utils/kinect_trained_ckpt_baseyou2me')
    total_step = len(data_loader)
    # print ("total iter", total_step)
    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
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

            if ((i + 1) % args.save_step == 0) or (i == total_step - 1):
                torch.save(decoder.state_dict(),
                           os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']  # Access the learning rate of the first parameter group
        print(f'Current Learning Rate: {current_lr:.6f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')

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
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=256, help='length of the pose/video sequences')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=5, help='step size for saving trained models')

    args = parser.parse_args()
    print(args)
    main(args)
