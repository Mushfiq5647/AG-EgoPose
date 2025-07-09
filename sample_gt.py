import numpy as np

# Full path to your .npy file
npy_path = "/data/My_Backup/UnrealEgo/scripts/data/UnrealEgoData/ArchVisInterior_ArchVis_RT/Day/rp_scott_rigged_005_ue4/SKM_MenReadingGlasses_Shape_01/023/Hip_Hop_Dancing__6_/all_data_with_img-256_hm-64_pose-16_npy/frame_0.npy"

# Load the file
data = np.load(npy_path, allow_pickle=True).item()

# Check available keys
print("Keys in the file:", list(data.keys()))

# Access gt_local_pose
if "gt_local_pose" in data:
    print("gt_local_pose shape:", data["gt_local_pose"].shape)
    print("gt_local_pose sample:\n", data["gt_local_pose"])
else:
    print("gt_local_pose not found in the file.")
