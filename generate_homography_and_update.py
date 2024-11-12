import os
import cv2
import numpy as np
from scipy.linalg import svd
import re
import shutil

def numerical_sort_key(filename):
    """
    Extract the numeric part of the filename for sorting purposes.
    Assumes filenames contain integers as part of their names.
    """
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')


def rename_images_gt_in_directory(directory, temp_dir, prefix):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get the list of image files and sort them numerically
    if prefix == 'img_':
        files = sorted([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))],
                             key=numerical_sort_key)
    else:
        files = sorted([f for f in os.listdir(directory) if f.endswith(('.txt'))],
                             key=numerical_sort_key)


    os.makedirs(temp_dir, exist_ok=True)

    # Copy and rename files to start from 1 in sequence
    for i, filename in enumerate(files, start=1):
        file_ext = os.path.splitext(filename)[1]  # Keep the original file extension
        new_name = f"{prefix}{i}{file_ext}"
        src_path = os.path.join(directory, filename)
        dst_path = os.path.join(temp_dir, new_name)
        shutil.copy2(src_path, dst_path)  # Use copy2 to preserve metadata if needed
        print(f"Copied and renamed '{filename}' to '{new_name}' in '{temp_dir}'")

    print("Renaming and copying complete! Renamed images are saved in 'temp_renamed'.")

def get_point_correspondences(img1, img2):
    # Detect and compute keypoints using ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Get the coordinates of the matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return pts1, pts2

def estimate_homography(pts1, pts2):
    assert pts1.shape == pts2.shape, "Point sets must have the same shape"

    N = pts1.shape[0]

    # Create the homogeneous linear system Ah = 0
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # Solve the homogeneous linear system via SVD
    _, _, vh = svd(A)
    h = vh[-1].reshape(3, 3)

    # Normalize the homography matrix
    h = h / h[0, 0]

    return h

#
# Example usage
image_directory = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/imgs'
temp_dir = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/img_dir_updated'
rename_images_gt_in_directory(image_directory, temp_dir,'img_')

gt_directory = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/ground_truth'
temp_gt_dir = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/ground_truth_updated'
rename_images_gt_in_directory(gt_directory, temp_gt_dir,'gt_')



# Example usage
image_dir = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/img_dir_updated'
homography_dir = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/binchen/kitchen/homography'

os.makedirs(homography_dir, exist_ok=True)

# Sort files numerically based on the numeric part in the filename
image_files = sorted(os.listdir(image_dir), key=numerical_sort_key)

for i in range(len(image_files) - 1):
    img1 = cv2.imread(os.path.join(image_dir, image_files[i]))
    img2 = cv2.imread(os.path.join(image_dir, image_files[i + 1]))

    pts1, pts2 = get_point_correspondences(img1, img2)
    H = estimate_homography(pts1, pts2)

    homography_file = os.path.join(homography_dir, f'h{i}.txt')
    np.savetxt(homography_file, H)

    print(f"Saved homography matrix between {image_files[i]} and {image_files[i + 1]} to {homography_file}")


