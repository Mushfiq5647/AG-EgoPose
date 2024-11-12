import json
import pickle
import os
import numpy as np

gt_path = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/mengyu_new/kitchen2/pseudo_gt.pkl'
gt_output_path = 'D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/mengyu_new/kitchen2/ground_truth/'

# Create the output folder if it doesn't exist
os.makedirs(gt_output_path, exist_ok=True)

# Load the .pkl file
with open(gt_path, 'rb') as f:
    data = pickle.load(f)

# with open('test_egopw.txt', 'w') as f:
#     f.write(str(data))

for i, frame_data in enumerate(data):
    # Extract "estimated_local_pose"
    optimised_pose = frame_data.get("optimized_local_pose")
    if optimised_pose is not None:
        # Convert to numpy array if necessary
        estimated_pose = np.array(optimised_pose)
        # Define output file path
        output_file_path = os.path.join(gt_output_path, f'gt_{i+169}.txt')

        # Save the array as a .txt file
        np.savetxt(output_file_path, estimated_pose, fmt="%.6f")
    else:
        print(f"Frame {i} does not contain 'estimated_local_pose' data.")

print("All frames with 'estimated_local_pose' have been saved.")
