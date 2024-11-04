import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

import numpy as np

frame_width, frame_height = 227, 227
num_frames_for_mhi = 10
decay_factor = 0.5

class SimpleMHIFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):  # 2 channels: intensity + flow magnitude
        super(SimpleMHIFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


def generate_mhi_for_frame(frames, decay=decay_factor):
    """Generate MHI for a target frame considering up to 10 past frames."""
    # Initialize the MHI
    mhi = np.zeros((frame_height, frame_width), dtype=np.float32)

    # Get the actual number of frames available
    num_available_frames = len(frames)

    # Process each pair in the frames (accumulating motion)
    for idx, (prev_frame, curr_frame) in enumerate(zip(frames[:-1], frames[1:])):
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute binary motion mask
        diff = cv2.absdiff(gray_prev, gray_curr)
        _, motion_mask = cv2.threshold(diff, 15, 1, cv2.THRESH_BINARY)

        # Update MHI with decay for older motion
        mhi = np.where(motion_mask == 1, num_frames_for_mhi - idx, mhi * decay)

    # Normalize MHI for visualization (optional)
    mhi = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(mhi)


def process_frame_sequence(frames_folder, current_index):
    start_index = max(1, current_index - num_frames_for_mhi)
    frames = []
    for i in range(start_index, current_index + 1):
        frame_path = os.path.join(frames_folder, f"imxx{i}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Frame {frame_path} not found.")
        frame = cv2.resize(frame, (frame_width, frame_height))
        frames.append(frame)

    # Generate MHI for the selected frame
    mhi = generate_mhi_for_frame(frames)
    return mhi

# Example usage:
def visualize_enhanced_mhi(mhi):
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

    ax1.imshow(mhi, cmap='gray')
    ax1.set_title('Motion Intensity')
    ax1.axis('off')

    plt.tight_layout()
    plt.show()
def estimate_mhi(frames_folder, current_frame_index):
    mhi = process_frame_sequence(frames_folder, current_frame_index)
    mhi_tensor = torch.tensor(mhi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    model = SimpleMHIFeatureExtractor()
    features = model(mhi_tensor)
    features = features.detach().cpu().numpy()
    return features
