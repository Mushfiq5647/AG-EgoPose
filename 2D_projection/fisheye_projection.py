#!/usr/bin/env python3
"""
3D to 2D projection using FishEye camera calibration
Based on the provided FishEyeCameraCalibrated class
"""
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from copy import deepcopy

class FishEyeCameraCalibrated:
    def __init__(self, calibration_file_path):
        with open(calibration_file_path) as f:
            calibration_data = json.load(f)
        self.intrinsic = np.array(calibration_data['intrinsic'])
        self.img_size = np.array(calibration_data['size'])  # w, h
        self.fisheye_polynomial = np.array(calibration_data['polynomialC2W'])
        self.fisheye_inverse_polynomial = np.array(calibration_data['polynomialW2C'])
        self.img_center = np.array([self.intrinsic[0][2], self.intrinsic[1][2]])

    def world2camera(self, point3D):
        """
        Project 3D world points to 2D camera coordinates
        point3D: numpy array of shape (N, 3) - [X, Y, Z] coordinates
        Returns: numpy array of shape (N, 2) - [u, v] pixel coordinates
        """
        point3D = deepcopy(point3D)
        
        # Debug: print original coordinates for first few points
        print("\n=== COORDINATE TRANSFORMATION DEBUG ===")
        print("Step 1 - Original coordinates (first 3 points):")
        for i in range(min(3, len(point3D))):
            print(f"  Point {i}: X={point3D[i,0]:.3f}, Y={point3D[i,1]:.3f}, Z={point3D[i,2]:.3f}")
        
        # Reorder axes to [X,Z,Y] order: X stays X, Z becomes Y, Y becomes Z
        temp_coords = point3D.copy()
        point3D[:, 0] = temp_coords[:, 0]  # X stays X
        point3D[:, 1] = temp_coords[:, 2]  # Z becomes Y
        point3D[:, 2] = temp_coords[:, 1]  # Y becomes Z

        print("\nStep 2 - After axis reordering [X,Z,Y] (first 3 points):")
        for i in range(min(3, len(point3D))):
            print(f"  Point {i}: X={point3D[i,0]:.3f}, Y={point3D[i,1]:.3f}, Z={point3D[i,2]:.3f}")
        
        # Apply flips
        point3D[:, 0] = point3D[:, 0] * 1   # Keep new X (was Z)
        point3D[:, 1] = point3D[:, 1] * -1  # Flip new Y (was X) - try flipping Y to correct orientation
        point3D[:, 2] = point3D[:, 2] * -1  # Flip new Z (was Y)
        
        print("\nStep 3 - After flips [X*1, Y*-1, Z*-1] (first 3 points):")
        for i in range(min(3, len(point3D))):
            print(f"  Point {i}: X={point3D[i,0]:.3f}, Y={point3D[i,1]:.3f}, Z={point3D[i,2]:.3f}")
        print("="*50)

        point3D = point3D.T  # Transpose to 3xN
        xc, yc = self.img_center[0], self.img_center[1]
        
        point2D_list = []
        
        for i in range(point3D.shape[1]):  # For each point
            x, y, z = point3D[0, i], point3D[1, i], point3D[2, i]
            
            norm = np.linalg.norm([x, y])
            
            if norm != 0:
                theta = np.arctan(z / norm)
                invnorm = 1.0 / norm
                t = theta
                rho = self.fisheye_inverse_polynomial[0]
                t_i = 1.0
                
                for j in range(1, len(self.fisheye_inverse_polynomial)):
                    t_i *= t
                    rho += t_i * self.fisheye_inverse_polynomial[j]
                
                u = x * invnorm * rho + xc
                v = y * invnorm * rho + yc
            else:
                u, v = xc, yc

            point2D_list.append([u, v])
        
        return np.array(point2D_list)

def draw_joints_on_image(image_path, joints_2d, joint_names=None, output_path='overlay_global.png'):
    """Draw 2D joints on image with labels"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Colors for different joint types
    colors = {
        'head': (0, 255, 255),      # Yellow
        'arm': (0, 0, 255),         # Red  
        'torso': (255, 0, 0),       # Blue
        'leg': (0, 255, 0),         # Green
        'default': (255, 255, 255)  # White
    }
    
    for i, (u, v) in enumerate(joints_2d.astype(int)):
        # Choose color based on joint name
        if joint_names and i < len(joint_names):
            joint_name = joint_names[i].lower()
            if 'head' in joint_name or 'neck' in joint_name:
                color = colors['head']
            elif 'shoulder' in joint_name or 'elbow' in joint_name or 'wrist' in joint_name:
                color = colors['arm']
            elif 'hip' in joint_name or 'torso' in joint_name:
                color = colors['torso']
            elif 'knee' in joint_name or 'ankle' in joint_name or 'foot' in joint_name:
                color = colors['leg']
            else:
                color = colors['default']
        else:
            color = colors['default']
            
        # Draw joint
        cv2.circle(img, (int(u), int(v)), 8, (0, 0, 0), -1)  # Black outline
        cv2.circle(img, (int(u), int(v)), 6, color, -1)      # Colored center
        
        # Add joint index/name
        label = str(i) if not joint_names else f"{i}:{joint_names[i][:3]}"
        cv2.putText(img, label, (int(u)+10, int(v)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(str(output_path), img)
    print(f"Saved overlay to: {output_path}")
    return img

def main():
    parser = argparse.ArgumentParser(description="Project 3D joints to 2D using fisheye camera")
    parser.add_argument('--calibration', type=str, default='2D_projection/fisheye_global.json',
                       help='Path to fisheye calibration JSON file')
    parser.add_argument('--image', type=str, default='/data/My_Backup/Dataset/TestDataset_EgocentricGlobalPose/jian1/imgs/img_000005.jpg',
                       help='Path to input image')
    parser.add_argument('--joints_3d', type=str,
                       help='Path to 3D joints numpy file (.npy) or provide as comma-separated values')
    parser.add_argument('--output', type=str, default='fisheye_overlay.png',
                       help='Output image path')
    parser.add_argument('--joint_names', type=str, nargs='*',
                       help='Optional joint names for labeling')
    
    args = parser.parse_args()
    
    # Load fisheye camera
    camera = FishEyeCameraCalibrated(args.calibration)
    print(f"Loaded fisheye camera: {camera.img_size[0]}x{camera.img_size[1]}")
    joints_3d = np.array([
        [0.97671735, 1.54327900, 0.19573466],
        [1.00835532, 1.49552154, 0.34671996],
        [1.23131206, 1.37301158, 0.50117951],
        [1.50747446, 1.41099229, 0.50216513],
        [0.99516824, 1.49045588, 0.05789975],
        [1.23308837, 1.36875226, -0.07309028],
        [1.51177879, 1.37484437, -0.07116388],
        [0.93624314, 0.89909887, 0.28257682],
        [0.97603754, 0.50483918, 0.35706449],
        [0.91807897, 0.09102662, 0.38360416],
        [1.09489666, 0.02784309, 0.44515897],
        [0.95348943, 0.90691681, 0.11581825],
        [0.97763022, 0.50964071, 0.05130759],
        [0.93084831, 0.09490796, 0.01795208],
        [1.11118573, 0.02787152, -0.02710544]
    ], dtype=np.float32)

    
    print(f"Loaded {len(joints_3d)} 3D joints")
    print(f"3D coordinates range:")
    print(f"  X: {joints_3d[:, 0].min():.3f} to {joints_3d[:, 0].max():.3f}")
    print(f"  Y: {joints_3d[:, 1].min():.3f} to {joints_3d[:, 1].max():.3f}")
    print(f"  Z: {joints_3d[:, 2].min():.3f} to {joints_3d[:, 2].max():.3f}")
    
    # Project to 2D
    joints_2d = camera.world2camera(joints_3d)
    
    # Check bounds
    W, H = camera.img_size[0], camera.img_size[1]
    in_bounds = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < W) & 
                 (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < H))
    
    print(f"Projected to 2D:")
    print(f"  U: {joints_2d[:, 0].min():.1f} to {joints_2d[:, 0].max():.1f}")
    print(f"  V: {joints_2d[:, 1].min():.1f} to {joints_2d[:, 1].max():.1f}")
    print(f"  In bounds: {in_bounds.sum()}/{len(joints_2d)}")
    
    # Draw joints on image
    img = draw_joints_on_image(args.image, joints_2d, args.joint_names, args.output)
    
    # Save 2D coordinates
    coords_file = Path(args.output).with_suffix('.npy')
    np.save(coords_file, joints_2d)
    print(f"Saved 2D coordinates to: {coords_file}")

if __name__ == '__main__':
    # Example usage if run directly
    import sys
    main()
