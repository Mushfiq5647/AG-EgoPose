import numpy as np
import matplotlib.pyplot as plt


gt_data = []
file_path = 'SceneEgo_train/train/diogo1/groundtruth_updated/gt_54.txt'
with open(file_path, 'r') as file:
    for line in file:
        # Remove any brackets or extra characters
        line = line.strip().replace('[', '').replace(']', '')

        # Convert the line into a list of floats and append it to data
        row = [float(value) for value in line.split()]
        gt_data.append(row)  # Each row is appended as a list

    # Flatten the 2D list into a 1D list
    egopose_gt = [value for row in gt_data for value in row]
joints = np.array(egopose_gt).reshape(15, 3)

# Define a joint hierarchy based on an assumed structure
connections = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
         (11, 12), (12, 13), (13, 14), (7, 11)]

# Plotting with connections
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the joints
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='blue')

#Draw connections
for connection in connections:
    start, end = connection
    ax.plot([joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]], color='red')

# Annotate each joint with its index
for i in range(joints.shape[0]):
    ax.text(joints[i, 0], joints[i, 1], joints[i, 2], f"{i}", color='red')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Joint Positions with Connections')

plt.show()
