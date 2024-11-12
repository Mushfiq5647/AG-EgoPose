import json
import os

oproot = 'you2me_ds_release_cmu/cmu'
with open(oproot + "/1-catch1/features/openpose/output_json/imxx1_keypoints.json", 'r') as f:
    js = json.loads(f.read())
    if 'people' not in js or len(js['people']) == 0:
        # No people detected, handle missing data
        pose2 = [0] * 75
    else:
        pose_keypoints = js['people'][0].get('pose_keypoints_2d', [])

        if len(pose_keypoints) == 0:
            # If no keypoints are found, set to a default value
            pose2 = [0] * 75
        else:
            pose2 = pose_keypoints
            third_person_pose = [pose2[i] for i in range(len(pose2)) if (i + 1) % 3 != 0]

with open('check.txt', 'w') as f:
    f.write(str(third_person_pose))

print(len(third_person_pose))