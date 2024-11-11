import json
import pickle
import os
import cv2

# Directories to save the images and ground truth data
image_save_dir = 'SceneEgo_test/jian2/image_dir'
gt_save_dir = 'SceneEgo_test/jian2/gt_dir'

# Create directories if they don't exist
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(gt_save_dir, exist_ok=True)


syn_path = 'SceneEgo_test/jian2/syn.json'
gt_path = 'SceneEgo_test/jian2/local_pose_gt.pkl'
img_data_path = 'SceneEgo_test/jian2/imgs'
with open(syn_path, 'r') as f:
    syn_data = json.load(f)

ego_start_frame = syn_data['ego']
ext_start_frame = syn_data['ext']

with open(gt_path, 'rb') as f:
    pose_gt_data = pickle.load(f)

image_path_list = []
gt_pose_list = []

for pose_gt_item in pose_gt_data:
    ext_id = pose_gt_item['ext_id']
    ego_pose_gt = pose_gt_item['ego_pose_gt']
    if ego_pose_gt is None:
        print("Yes")
        continue
    ego_id = ext_id - ext_start_frame + ego_start_frame
    egocentric_image_name = "img_%06d.jpg" % ego_id

    image_path = os.path.join(img_data_path, egocentric_image_name)
    if not os.path.exists(image_path):
        continue
    image_path_list.append(image_path)
    gt_pose_list.append(ego_pose_gt)
    # Load the image from image_path
    image = cv2.imread(image_path)

    # Save the image to the designated directory
    image_filename = os.path.join(image_save_dir, f"img_{ego_id}.jpg")
    cv2.imwrite(image_filename, image)
    # Save the corresponding ground truth to a text file
    gt_filename = os.path.join(gt_save_dir, f"gt_{ego_id}.txt")
    with open(gt_filename, 'w') as gt_file:
        gt_file.write(str(ego_pose_gt))
print("dataset length: {}".format(len(image_path_list)))
print("GT length: {}".format(len(gt_pose_list)))

with open('syn_image.txt', 'w') as f:
    f.write(f"{image_path_list}\n")
