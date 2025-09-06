#!/usr/bin/env python3
"""
Script to analyze heatmap .npy files and check if they contain proper Gaussian blobs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import ndimage
from scipy.stats import multivariate_normal

def analyze_heatmap_blob(heatmap, joint_idx=0, threshold=0.1):
    """
    Analyze a single joint heatmap to check if it's a proper Gaussian blob
    
    Args:
        heatmap: 2D numpy array (H, W) for single joint
        joint_idx: Joint index for labeling
        threshold: Minimum value to consider as "signal"
    
    Returns:
        dict: Analysis results
    """
    results = {
        'joint_idx': joint_idx,
        'shape': heatmap.shape,
        'min': heatmap.min(),
        'max': heatmap.max(),
        'mean': heatmap.mean(),
        'std': heatmap.std(),
        'has_peak': False,
        'peak_location': None,
        'blob_size': 0,
        'is_gaussian_like': False,
        'peak_value': 0.0
    }
    
    # Basic statistics
    if heatmap.max() > threshold:
        results['has_peak'] = True
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        results['peak_location'] = peak_idx
        results['peak_value'] = heatmap[peak_idx]
        
        # Measure blob size (connected components above threshold)
        binary_mask = heatmap > threshold
        labeled, num_features = ndimage.label(binary_mask)
        
        if num_features > 0:
            # Find the component containing the peak
            peak_label = labeled[peak_idx]
            blob_mask = (labeled == peak_label)
            results['blob_size'] = np.sum(blob_mask)
            
            # Check if it's roughly Gaussian by measuring falloff from peak
            center_y, center_x = peak_idx
            h, w = heatmap.shape
            
            # Sample values at different distances from center
            distances = []
            values = []
            
            for r in range(1, min(center_y, center_x, h-center_y, w-center_x)):
                # Sample 8 points around the circle at distance r
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                for angle in angles:
                    y = int(center_y + r * np.sin(angle))
                    x = int(center_x + r * np.cos(angle))
                    if 0 <= y < h and 0 <= x < w:
                        distances.append(r)
                        values.append(heatmap[y, x])
            
            if len(distances) > 0:
                distances = np.array(distances)
                values = np.array(values)
                
                # Check if values decrease with distance (Gaussian property)
                unique_distances = np.unique(distances)
                if len(unique_distances) > 2:
                    avg_values_by_distance = []
                    for d in unique_distances:
                        avg_val = np.mean(values[distances == d])
                        avg_values_by_distance.append(avg_val)
                    
                    # Check if mostly decreasing
                    decreasing_count = 0
                    for i in range(1, len(avg_values_by_distance)):
                        if avg_values_by_distance[i] <= avg_values_by_distance[i-1]:
                            decreasing_count += 1
                    
                    # Consider Gaussian-like if mostly decreasing
                    results['is_gaussian_like'] = (decreasing_count / (len(avg_values_by_distance) - 1)) > 0.7
    
    return results

def visualize_heatmap(heatmap, title="Heatmap", joint_idx=None):
    """Visualize a single heatmap"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original heatmap
    im1 = axes[0].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[0].set_title(f'{title} - Original')
    axes[0].set_xlabel(f'Min: {heatmap.min():.3f}, Max: {heatmap.max():.3f}')
    plt.colorbar(im1, ax=axes[0])
    
    # Thresholded view (>0.1)
    thresh_map = np.where(heatmap > 0.1, heatmap, 0)
    im2 = axes[1].imshow(thresh_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'{title} - Threshold > 0.1')
    plt.colorbar(im2, ax=axes[1])
    
    # Profile through peak
    if heatmap.max() > 0.1:
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Horizontal profile through peak
        h_profile = heatmap[peak_y, :]
        axes[2].plot(h_profile, 'b-', label='Horizontal', linewidth=2)
        
        # Vertical profile through peak  
        v_profile = heatmap[:, peak_x]
        axes[2].plot(v_profile, 'r-', label='Vertical', linewidth=2)
        
        axes[2].axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Threshold')
        axes[2].set_title(f'Profiles through Peak ({peak_y}, {peak_x})')
        axes[2].set_xlabel('Pixel Position')
        axes[2].set_ylabel('Heatmap Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No significant peak found', 
                    transform=axes[2].transAxes, ha='center', va='center')
        axes[2].set_title('No Peak Analysis Available')
    
    plt.tight_layout()
    return fig

def main():
    print("🔍 Heatmap Blob Analyzer")
    print("=" * 50)
    
    # Find heatmap files
    heatmap_patterns = [
        "../**/heatmap/*.npy",
        "../../**/heatmap/*.npy", 
        "../**/heatmap_*.npy",
        "../../**/heatmap_*.npy"
    ]
    
    heatmap_files = []
    for pattern in heatmap_patterns:
        heatmap_files.extend(glob.glob(pattern, recursive=True))
    
    if not heatmap_files:
        print("❌ No heatmap .npy files found!")
        print("   Searched patterns:", heatmap_patterns)
        
        # Try current directory
        local_files = glob.glob("*.npy") + glob.glob("heatmap*.npy")
        if local_files:
            print(f"   Found {len(local_files)} .npy files in current directory:")
            for f in local_files[:5]:  # Show first 5
                print(f"     - {f}")
            heatmap_files = local_files
        else:
            return
    
    print(f"📁 Found {len(heatmap_files)} heatmap files")
    
    # Analyze first few files
    for i, file_path in enumerate(heatmap_files[:3]):
        print(f"\n📊 Analyzing: {os.path.basename(file_path)}")
        print("-" * 40)
        
        try:
            # Load heatmap
            heatmap_data = np.load(file_path)
            print(f"   Shape: {heatmap_data.shape}")
            print(f"   Dtype: {heatmap_data.dtype}")
            print(f"   Range: [{heatmap_data.min():.4f}, {heatmap_data.max():.4f}]")
            print(f"   Mean: {heatmap_data.mean():.4f}")
            
            # Handle different shapes
            if heatmap_data.ndim == 2:
                # Single joint heatmap (H, W)
                results = analyze_heatmap_blob(heatmap_data, joint_idx=0)
                print(f"   Has Peak: {results['has_peak']}")
                if results['has_peak']:
                    print(f"   Peak Location: {results['peak_location']}")
                    print(f"   Peak Value: {results['peak_value']:.4f}")
                    print(f"   Blob Size: {results['blob_size']} pixels")
                    print(f"   Gaussian-like: {results['is_gaussian_like']}")
                
                # Visualize
                fig = visualize_heatmap(heatmap_data, f"File {i+1}", 0)
                save_path = f"heatmap_analysis_{i+1}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   💾 Saved visualization: {save_path}")
                
            elif heatmap_data.ndim == 3:
                # Multi-joint heatmap (J, H, W)
                num_joints = heatmap_data.shape[0]
                print(f"   Number of Joints: {num_joints}")
                
                good_joints = 0
                for j in range(num_joints):
                    joint_hm = heatmap_data[j]
                    results = analyze_heatmap_blob(joint_hm, joint_idx=j)
                    
                    if results['has_peak']:
                        good_joints += 1
                        if j < 3:  # Show details for first 3 joints
                            print(f"     Joint {j}: Peak at {results['peak_location']}, "
                                  f"Value={results['peak_value']:.3f}, "
                                  f"Gaussian={results['is_gaussian_like']}")
                
                print(f"   ✅ {good_joints}/{num_joints} joints have significant peaks")
                
                # Visualize first 3 joints
                if num_joints > 0:
                    fig, axes = plt.subplots(1, min(3, num_joints), figsize=(15, 5))
                    if num_joints == 1:
                        axes = [axes]
                    
                    for j in range(min(3, num_joints)):
                        im = axes[j].imshow(heatmap_data[j], cmap='hot', interpolation='nearest')
                        axes[j].set_title(f'Joint {j}')
                        plt.colorbar(im, ax=axes[j])
                    
                    plt.tight_layout()
                    save_path = f"multi_joint_analysis_{i+1}.png"
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"   💾 Saved multi-joint visualization: {save_path}")
            
            else:
                print(f"   ⚠️  Unexpected shape: {heatmap_data.shape}")
                
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
    
    print(f"\n✅ Analysis complete! Check the generated .png files for visualizations.")

if __name__ == "__main__":
    main()
