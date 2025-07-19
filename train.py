# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import torch
import torch.nn as nn
import tqdm
from options.train_options import TrainOptions
import numpy as np
import os
import pickle
from utils.data_loader import dataloader_full
from utils.build_annotation import Annotation
from action_recognition import ActionFormerFeatureExtractor
from action_recognition import initialize_actionformer
from utils.model import SpatioTemporalTransformer, FeatureDecoder, FeatureEncoder
from utils.loss import LossFuncLimb, LossFuncMPJPE, LossFuncCosSim  # Add bone length loss import
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import MultiStepLR
from utils.build_annotation import Annotation
from torchvision import transforms
torch.set_printoptions(threshold=torch.inf)
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OS:", os.cpu_count())
print("Working On:",device)
# Number of joints and coordinates per joint
num_joints = 16
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

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight,
            a=0,
            nonlinearity='relu'
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def check_nan(tensor, name, epoch, i):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}! Epoch: {epoch}, Iteration: {i}")
        return True
    return False

def update_learning_rate(self):
    old_lr = self.optimizers[0].param_groups[0]['lr']
    for scheduler in self.schedulers:
        scheduler.step()
    lr = self.optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

def main(args):
    actionformer_feature_extractor = initialize_actionformer(config_file_path=args.config_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    # Create loss log file
    loss_log_path = 'loss_log.txt'
    with open(loss_log_path, 'w') as f:
        f.write("Epoch,Train_MPJPE, Train_Cos, Train_Bone, Train_Total, Val_MPJPE, Val_Cos,Val_Bone, Val_Total\n")
    
    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

    # with open(args.annotation_path, 'rb') as f:
    #     annotation = pickle.load(f)
    opt = TrainOptions().parse()
    data_loader = dataloader_full(opt, transform, mode='train')
    val_loader = dataloader_full(opt, transform, mode='validation')
    print("Data Loading complete", len(data_loader))
    print("Total dataset", len(data_loader.dataset))
    print("Validation Data Loading complete", len(val_loader))
    print("Total validation dataset", len(val_loader.dataset))
    encoder = FeatureEncoder(args.embed_feature_dim, actionformer_feature_extractor).to(device)
    spatio_temporal_transformer = SpatioTemporalTransformer(
                         args.embed_feature_dim,
                         args.num_layers).to(device)


    decoder = FeatureDecoder(args.hidden_size, args.seq_length,
                     spatio_temporal_transformer).to(device)

    encoder.apply(initialize_weights)
    decoder.apply(initialize_weights)

    # loss and optimizer
    criterion = nn.MSELoss()
    limb_loss_func = LossFuncLimb().to(device)  # Use existing bone length loss
    mpjpe_loss_func = LossFuncMPJPE().to(device)
    cos_sim_loss_func = LossFuncCosSim().to(device)
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=5e-4
        )
    model_dir = os.path.abspath('./utils/kinect_trained_ckpt_final')
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        print("Printing epoch:", epoch)
        encoder.train()
        decoder.train()
        
        # Initialize training loss accumulators
        train_mpjpe_losses = []
        train_cos_losses = []
        train_bone_losses = []
        train_total_losses = []
        
        for i, (batch) in enumerate(data_loader):
            print("Printing iteration number:", i)
            # Instead of: for i, (images, homography, gt_egoposes, lengths) in enumerate(data_loader)
            images = batch['input_rgb_left'].to(device)  # Tensor
            homography = batch['input_homography'].to(device)  # Tensor
            gt_egoposes = batch['gt_local_pose'].to(device)  # Tensor
            gt_egoposes = gt_egoposes/10
            # lengths = batch['window_size']  # int
            images = images.to(device)
            B = images.size(0)
            print("Image", images.shape)
            print("Homography", homography.shape)
            print("Images shape:", images.shape)
            img_min = images.min().item()
            img_max = images.max().item()
            has_nan = torch.isnan(images).any().item()
            has_inf = torch.isinf(images).any().item()
            print(f"Batch {i}: image.range=({img_min:.4f}, {img_max:.4f}), NaN={has_nan}, Inf={has_inf}")
            if has_nan or has_inf:
                raise ValueError(f"Invalid pixels detected in batch {i}")
            features = encoder(images)
            print("Features before decoder shape:", features.shape)
            lengths = args.seq_length
            final = decoder(features, lengths, homography)
            print("GT Poses", gt_egoposes.shape)
            print("Gt Poses min/max:", gt_egoposes.min().item(), gt_egoposes.max().item())
            print("Outputs Final", final.shape)

            # Reshape for loss computation: (B*T, num_joints, 3)
            B, T = final.shape[:2]
            final_reshaped = final.reshape(B*T, num_joints, 3)
            gt_reshaped = gt_egoposes.reshape(B*T, num_joints, 3)

            # Compute multiple losses
            # mse_loss = criterion(final, gt_egoposes)
            mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
            bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
            cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)

            # Combined loss with weights from options
            final_loss = (
                         opt.lambda_mpjpe * mpjpe_loss + 
                         opt.lambda_cos_sim * cos_loss + 
                         opt.lambda_bone_length * bone_length_loss)
            
            # Store losses for epoch averaging
            train_mpjpe_losses.append(mpjpe_loss.item())
            train_cos_losses.append(cos_loss.item())
            train_bone_losses.append(bone_length_loss.item())
            train_total_losses.append(final_loss.item())
            
            print(f"MPJPE: {mpjpe_loss:.4f}, "
                  f"Cos: {cos_loss.item():.4f}, Bone: {bone_length_loss.item():.4f}")

            # Backward and optimize with mixed precision
            decoder.zero_grad()
            encoder.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_value)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_value)
            optimizer.step()
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, final_loss.item(), np.exp(final_loss.item())))
                # torch.save(decoder.state_dict(), os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                # torch.save(encoder.state_dict(), os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

            if ((i + 1) % args.save_step == 0) or (i == total_step - 1):
                torch.save(decoder.state_dict(),
                           os.path.join(model_dir, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(model_dir, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        
        # Calculate training epoch averages
        train_avg_mpjpe = np.mean(train_mpjpe_losses)
        train_avg_cos = np.mean(train_cos_losses)
        train_avg_bone = np.mean(train_bone_losses)
        train_avg_total = np.mean(train_total_losses)
        
        encoder.eval()
        decoder.eval()

        val_mpjpe_losses = []
        val_cos_losses = []
        val_bone_losses = []
        val_total_losses = []
        
        with torch.no_grad():
            for j, val_batch in enumerate(val_loader):
                images = val_batch['input_rgb_left'].to(device)
                homography = val_batch['input_homography'].to(device)
                gt_egoposes = val_batch['gt_local_pose'].to(device)
                gt_egoposes = gt_egoposes/10
                features = encoder(images)
                lengths = args.seq_length
                final = decoder(features, lengths, homography)
                
                # Compute validation losses
                B, T = final.shape[:2]
                final_reshaped = final.reshape(B*T, num_joints, 3)
                gt_reshaped = gt_egoposes.reshape(B*T, num_joints, 3)

                mpjpe_loss = mpjpe_loss_func(final_reshaped, gt_reshaped)
                bone_length_loss = limb_loss_func(final_reshaped, gt_reshaped)
                cos_loss = cos_sim_loss_func(final_reshaped, gt_reshaped)
                
                val_loss = (
                           opt.lambda_mpjpe * mpjpe_loss + 
                           opt.lambda_cos_sim * cos_loss + 
                           opt.lambda_bone_length * bone_length_loss)
                
                # Store validation losses
                val_mpjpe_losses.append(mpjpe_loss.item())
                val_cos_losses.append(cos_loss.item())
                val_bone_losses.append(bone_length_loss.item())
                val_total_losses.append(val_loss.item())

                if j % args.log_step == 0:
                    print(f"[VAL] Epoch [{epoch}/{args.num_epochs}] Step [{j}/{len(val_loader)}] "
                          f"Loss: {val_loss.item():.4f}")

        # Calculate validation epoch averages
        val_avg_mpjpe = np.mean(val_mpjpe_losses) if val_mpjpe_losses else 0.0
        val_avg_cos = np.mean(val_cos_losses) if val_cos_losses else 0.0
        val_avg_bone = np.mean(val_bone_losses) if val_bone_losses else 0.0
        val_avg_total = np.mean(val_total_losses) if val_total_losses else 0.0
        
        # Print epoch summary
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Train - MPJPE: {train_avg_mpjpe:.4f}, Cos: {train_avg_cos:.4f}, Bone: {train_avg_bone:.4f}, Total: {train_avg_total:.4f}")
        print(f"Val   - MPJPE: {val_avg_mpjpe:.4f}, Cos: {val_avg_cos:.4f}, Bone: {val_avg_bone:.4f}, Total: {val_avg_total:.4f}")
        
        # Save losses to file
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch},{train_avg_mpjpe:.6f},{train_avg_cos:.6f},{train_avg_bone:.6f},{train_avg_total:.6f},"
                   f"{val_avg_mpjpe:.6f},{val_avg_cos:.6f},{val_avg_bone:.6f},{val_avg_total:.6f}\n")

        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print("New learning rate:", new_lr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='actionformer/config/ego4D_egovlp.yaml', help='path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--annotation_path', type=str, required=True, help='path for annotation wrapper')

    parser.add_argument('--image_dir', type=str, default='/data/My_Backup/UnrealEgo/scripts/data/UnrealEgoData',
                        help='directory for resized images')
    parser.add_argument('--h_dir', type=str, default='D:/Dataset/EgoPW_dataset/EgoPW_dataset_release',
                        help='directory for resized images')
    parser.add_argument('--openpose_dir', type=str, default='you2me_ds_release_kinect/kinect',
                        help='directory for resized images')

    parser.add_argument('--embed_feature_dim', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=64, help='length of the pose/video sequences')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--log_step', type=int, default=20, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=20, help='step size for saving trained models')

    args = parser.parse_args()
    print(args)
    main(args)
