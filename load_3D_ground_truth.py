# import pickle as pkl
# import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2
#
# base_dir = '/data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/ayush/out'
# pkl_file = '/data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/ayush/out/pseudo_gt.pkl'
#
# with open(pkl_file, 'rb') as f:
#     data = pkl.load(f)
#
# image_name = data[0]['image_name']
# keypoints_2d = data[0]['joints_2d']
# keypoints_3d = data[0]['optimized_local_pose']
# print(keypoints_3d)
#
# img_path = os.path.join(base_dir,'imgs', image_name)
# print(img_path)
# # 3) Read & sanity‐check
# image = cv2.imread(img_path)
# if image is None:
#     raise FileNotFoundError(f"Couldn’t load image at {img_path}")
#
# # 4) Convert to RGB for Matplotlib
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #
# # # 5) Plot it
# plt.figure(figsize=(8,8))
# plt.imshow(image)
# plt.scatter(keypoints_2d[:,0], keypoints_2d[:,1],
#             c='r', s=30, marker='o')
# plt.axis('off')
# plt.title("2D GT Keypoints")
# plt.show()
#
#
#!/usr/bin/env python3
from __future__ import annotations
import json
import argparse
import pickle
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------- skeleton spec (15 joints) -----------------
list_joints = [
    "Neck", "Right_shoulder", "Right_elbow", "Right_wrist",
    "Left_shoulder", "Left_elbow", "Left_wrist",
    "Right_hip", "Right_knee", "Right_ankle", "Right_foot",
    "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
]
kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
lines = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11),
    (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14), (7, 11)
]

def load_extrinsics_from_json(json_path: Path):
    with open(json_path, "r") as f:
        calib = json.load(f)
    T = np.array(calib["extrinsic"], dtype=np.float64)  # (4,4)
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def world_to_camera_single(X_world: np.ndarray, R: np.ndarray, t: np.ndarray, is_cam_to_world: bool = True):
    """
    If (R,t) are camera->world (R_wc, t_wc), set is_cam_to_world=True (default).
    If they are world->camera (R_cw, t_cw), set is_cam_to_world=False.
    """
    if is_cam_to_world:
        R_cw = R.T
        t_cw = -R.T @ t
    else:
        R_cw, t_cw = R, t
    return (X_world @ R_cw.T) + t_cw  # (J,3)

def axes_test_to_train(X_cam_test: np.ndarray):
    """
    Test axes:  (depth, height, side)  ->  Train axes: (side, depth, height-down)
    (X_T, Y_T, Z_T) = (Z_S, -X_S, -Y_S)
    """
    Xs, Ys, Zs = X_cam_test[:, 0], X_cam_test[:, 1], X_cam_test[:, 2]
    return np.stack([Zs, -Xs, -Ys], axis=1)


def to_J3(arr: np.ndarray, frame: int | None) -> np.ndarray:
    """Coerce common shapes to (J,3)."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 3:          # (N,J,3) / (1,J,3)
        idx = 0 if frame is None else frame
        a = a[idx]
    elif a.ndim == 2 and a.shape[-1] == 3:        # (J,3)
        pass
    elif a.ndim == 2 and a.shape[1] % 3 == 0:     # (N,3J)
        idx = 0 if frame is None else frame
        a = a[idx].reshape(-1, 3)
    elif a.ndim == 1 and a.size % 3 == 0:         # (3J,)
        a = a.reshape(-1, 3)
    else:
        raise ValueError(f"Unhandled pose array shape: {a.shape}")
    return a.astype(np.float32)

def set_equal_3d(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    if max_range == 0: max_range = 1.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

def draw_skeleton(ax, pose: np.ndarray, use_parents: bool = True):
    X, Y, Z = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.scatter(X, Y, Z, s=20)
    for i, (x, y, z) in enumerate(pose):
        name = list_joints[i] if i < len(list_joints) else str(i)
        ax.text(x, y, z, f"{i}:{name}", fontsize=8)
    if use_parents:
        for i, p in enumerate(kinematic_parents):
            if i == p: continue
            if i < len(pose) and p < len(pose):
                ax.plot([pose[i,0], pose[p,0]], [pose[i,1], pose[p,1]], [pose[i,2], pose[p,2]])
    else:
        for i, j in lines:
            if i < len(pose) and j < len(pose):
                ax.plot([pose[i,0], pose[j,0]], [pose[i,1], pose[j,1]], [pose[i,2], pose[j,2]])

def load_pose_from_mat(mat_path: Path, key: str, frame: int | None) -> np.ndarray:
    md = sio.loadmat(str(mat_path))
    if key not in md:
        raise KeyError(f"Key '{key}' not in {mat_path}. Keys: {list(md.keys())}")
    return to_J3(md[key], frame)

def load_pose_from_h5(h5_path: Path, index: int, dataset: str = "Annot3D") -> np.ndarray:
    import h5py
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset '{dataset}' not in {h5_path}. Keys: {list(f.keys())}")
        arr = np.array(f[dataset][index])  # could be (J,3), (3J,), (1,J,3), etc.
    # Coerce to (J,3)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        pose = arr
    elif arr.ndim == 1 and arr.size % 3 == 0:
        pose = arr.reshape(-1, 3)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        pose = arr[0]
    elif arr.ndim == 2 and arr.shape[0] == 3:
        pose = arr.T
    elif arr.ndim == 2 and arr.shape[1] % 3 == 0:
        pose = arr.reshape(-1, 3)
    else:
        raise ValueError(f"Unhandled Annot3D shape: {arr.shape}")
    return pose.astype(np.float32)

def load_pose_from_sceneego_pkl(annotation_pkl_path: Path, index: int, key: str = "ego_pose_gt") -> np.ndarray:
    with open(annotation_pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of dicts in {annotation_pkl_path}, got {type(data)}")
    if not (0 <= index < len(data)):
        raise IndexError(f"index {index} out of range for annotation with {len(data)} items")

    item = data[index]
    if key not in item:
        raise KeyError(f"Key '{key}' not found in annotation item. Keys: {list(item.keys())}")

    pose = np.asarray(item[key])
    # coerce to (J,3)
    if pose.ndim == 2 and pose.shape[-1] == 3:
        pass
    elif pose.ndim == 1 and pose.size % 3 == 0:
        pose = pose.reshape(-1, 3)
    elif pose.ndim == 3 and pose.shape[-1] == 3:
        pose = pose[0]
    else:
        raise ValueError(f"Unhandled pose shape from annotation: {pose.shape}")

    return pose.astype(np.float32)

def load_pose_from_egogta_pkl(annotation_pkl_path: Path, index: int, key: str = "joint_3d_local") -> np.ndarray:
    with open(annotation_pkl_path, "rb") as f:
        data = pickle.load(f)
    # if not isinstance(data, list):
    #     raise TypeError(f"Expected a list of dicts in {annotation_pkl_path}, got {type(data)}")
    # if not (0 <= index < len(data)):
    #     raise IndexError(f"index {index} out of range for annotation with {len(data)} items")

    item = data
    if key not in item:
        raise KeyError(f"Key '{key}' not found in annotation item. Keys: {list(item.keys())}")

    pose = np.asarray(item[key])
    # coerce to (J,3)
    if pose.ndim == 2 and pose.shape[-1] == 3:
        pass
    elif pose.ndim == 1 and pose.size % 3 == 0:
        pose = pose.reshape(-1, 3)
    elif pose.ndim == 3 and pose.shape[-1] == 3:
        pose = pose[0]
    else:
        raise ValueError(f"Unhandled pose shape from annotation: {pose.shape}")

    return pose.astype(np.float32)

def load_pose_from_global_pkl(pkl_path: Path, index: int) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Check if it's a list of dictionaries (like annotation.pkl format)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if not (0 <= index < len(data)):
            raise IndexError(f"index {index} out of range for list with {len(data)} items")
        
        item = data[index]
        
        # Look for pose-related keys
        pose_keys = ['ego_pose_gt', 'pose_gt', 'optimized_local_pose', 'pose']
        pose_key = None
        for key in pose_keys:
            if key in item:
                pose_key = key
                break
        
        if pose_key is None:
            raise KeyError(f"No pose key found in item. Available keys: {list(item.keys())}")
        
        pose = np.asarray(item[pose_key])
        
        # Handle different pose shapes
        if pose.ndim == 2 and pose.shape[-1] == 3:
            pass  # Already (J,3)
        elif pose.ndim == 1 and pose.size % 3 == 0:
            pose = pose.reshape(-1, 3)
        elif pose.ndim == 3 and pose.shape[-1] == 3:
            pose = pose[0]  # Take first frame if multiple
        else:
            raise ValueError(f"Unhandled pose shape from {pose_key}: {pose.shape}")
    
    # Handle numpy array formats
    else:
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(f"index {index} out of range for PKL with {arr.shape[0]} frames")
            pose = arr[index]
        elif arr.ndim == 2 and arr.shape[1] % 3 == 0:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(f"index {index} out of range for PKL with {arr.shape[0]} frames")
            pose = arr[index].reshape(-1, 3)
        elif arr.ndim == 2 and arr.shape[-1] == 3:
            pose = arr
        elif arr.ndim == 1 and arr.size % 3 == 0:
            pose = arr.reshape(-1, 3)
        else:
            raise ValueError(f"Unhandled PKL pose array shape: {arr.shape}")

    return pose.astype(np.float32)


def load_pose_from_pseudo_pkl(pkl_path: Path, index: int) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Check if it's a list of dictionaries (like annotation.pkl format)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if not (0 <= index < len(data)):
            raise IndexError(f"index {index} out of range for list with {len(data)} items")

        item = data[index]

        # Look for pose-related keys
        pose_keys = ['ego_pose_gt', 'pose_gt', 'optimized_local_pose', 'pose']
        pose_key = None
        for key in pose_keys:
            if key in item:
                pose_key = key
                break

        if pose_key is None:
            raise KeyError(f"No pose key found in item. Available keys: {list(item.keys())}")

        pose = np.asarray(item[pose_key])

        # Handle different pose shapes
        if pose.ndim == 2 and pose.shape[-1] == 3:
            pass  # Already (J,3)
        elif pose.ndim == 1 and pose.size % 3 == 0:
            pose = pose.reshape(-1, 3)
        elif pose.ndim == 3 and pose.shape[-1] == 3:
            pose = pose[0]  # Take first frame if multiple
        else:
            raise ValueError(f"Unhandled pose shape from {pose_key}: {pose.shape}")

    # Handle numpy array formats
    else:
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(f"index {index} out of range for PKL with {arr.shape[0]} frames")
            pose = arr[index]
        elif arr.ndim == 2 and arr.shape[1] % 3 == 0:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(f"index {index} out of range for PKL with {arr.shape[0]} frames")
            pose = arr[index].reshape(-1, 3)
        elif arr.ndim == 2 and arr.shape[-1] == 3:
            pose = arr
        elif arr.ndim == 1 and arr.size % 3 == 0:
            pose = arr.reshape(-1, 3)
        else:
            raise ValueError(f"Unhandled PKL pose array shape: {arr.shape}")

    return pose.astype(np.float32)

# '/data/My_Backup/Dataset/Mocap2-20231112T183152Z-001/Mocap2/test_data/test_data/weipeng_studio/weipeng_studio_gt'
# '/data/My_Backup/Dataset/Mocap2-20231112T183152Z-001/Mocap2/training_data/mo2cap2_chunk_0001.hdf5'
# '/data/My_Backup/Dataset/TestDataset_EgocentricGlobalPose/jian1/jian1.pkl'
# /data/My_Backup/Dataset/SceneEgo_test/diogo1/annotation.pkl
# /data/My_Backup/Dataset/EgoPW_dataset/EgoPW_original/ayush/kitchen2/pseudo_gt.pkl
#/home/mushfiq/Downloads/EgoGTA/EgoGTA_network_input/2020-05-21-13-54-43/0.pkl

def main():
    ap = argparse.ArgumentParser(description="Plot 3D pose from .mat or .hdf5 as a skeleton (interactive)")
    # src = ap.add_mutually_exclusive_group(required=True)
    ap.add_argument("--mat", type=str, help="Path to .mat (expects key pose_gt by default)")
    ap.add_argument("--h5", type=str,  help= "Path to .h5/.hdf5 (expects dataset Annot3D)")
    ap.add_argument("--egoglobal", type=str, help= "Path to pkl (expects dataset Annot3D)")
    ap.add_argument("--sceneego", type=str, help= "Path to pkl (expects dataset Annot3D)")
    ap.add_argument("--egopw", type=str, help= "Path to pkl (expects dataset Annot3D)")
    ap.add_argument("--egogta", type=str, help= "Path to pkl (expects dataset Annot3D)")
    ap.add_argument("--calib_json", type=str, default='fisheye_calibration.json',
                    help="Path to calibration JSON containing 4x4 'extrinsic'. If omitted, identity is used.")
    ap.add_argument("--extrinsics_format", type=str, default="cam2world",
                    choices=["cam2world", "world2camera"],
                    help="Interpretation of the JSON 'extrinsic'. Default: camera->world.")

    ap.add_argument("--key", default="pose_gt", type=str, help="Key in .mat (default: pose_gt)")
    ap.add_argument("--index", default=0, type=int, help="Frame index for HDF5 or batched arrays")
    ap.add_argument("--dataset", default="Annot3D", type=str, help="Dataset name in HDF5 (default: Annot3D)")
    ap.add_argument("--mm-to-m", action="store_true", help="Divide by 1000 (if Annot3D is in millimeters)")
    ap.add_argument("--use-lines", action="store_true", help="Use explicit edges instead of kinematic parents")
    args = ap.parse_args()

    if args.mat:
        pose = load_pose_from_mat(Path(args.mat), key=args.key, frame=args.index)
        print("In mat")
        pose = pose [:,[0,2,1]]
        pose[:, 0] *= -1.0
        pose[:, 2] *= -1.0
        pose = pose / 1000
    elif args.h5:
        print("In h5")
        pose = load_pose_from_h5(Path(args.h5), index=args.index, dataset=args.dataset)
        pose = pose / 1000.0

    elif args.sceneego:
        print("In sceneego")
        pose = load_pose_from_sceneego_pkl(Path(args.sceneego), index=args.index)

    elif args.egogta:
        print("In egogta")
        pose = load_pose_from_egogta_pkl(Path(args.egogta), index=args.index)

    elif args.egoglobal:
        print("In egoglobal")
        pose = load_pose_from_global_pkl(Path(args.egoglobal), index=args.index).astype(np.float32)
        if args.calib_json is not None:
            R, t = load_extrinsics_from_json(Path(args.calib_json))
            is_cam_to_world = (args.extrinsics_format == "cam2world")
        else:
            R, t = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            is_cam_to_world = True  # identity either way
        pose_cam = world_to_camera_single(pose, R, t, is_cam_to_world=is_cam_to_world)
        pose = axes_test_to_train(pose_cam)
        # pose = pose [:,[2,0,1]]
        # pose[:, 1] *= -1.0
        # pose[:, 2] *= -1.0

    else:
        print("In pseudo_gt")
        pose = load_pose_from_pseudo_pkl(Path(args.egopw), index=args.index).astype(np.float32)



    pose = pose - pose[0]
    print(pose)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_skeleton(ax, pose, use_parents=not args.use_lines)
    X, Y, Z = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_equal_3d(ax, X, Y, Z)
    ax.view_init(elev=20, azim=35)
    ax.grid(True)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

