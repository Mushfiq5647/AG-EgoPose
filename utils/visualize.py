# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_joints(joints, ax):
    # Spine
    ax.plot3D(joints[[0, 1], 0], joints[[0, 1], 2], joints[[0, 1], 1])  # SpineBase -> SpineMid
    ax.plot3D(joints[[1, 2], 0], joints[[1, 2], 2], joints[[1, 2], 1])  # SpineMid -> Neck
    ax.plot3D(joints[[2, 3], 0], joints[[2, 3], 2], joints[[2, 3], 1])  # Neck -> Head

    # Left arm
    ax.plot3D(joints[[2, 4], 0], joints[[2, 4], 2], joints[[2, 4], 1])  # Neck -> ShoulderLeft
    ax.plot3D(joints[[4, 5], 0], joints[[4, 5], 2], joints[[4, 5], 1])  # ShoulderLeft -> ElbowLeft
    ax.plot3D(joints[[5, 6], 0], joints[[5, 6], 2], joints[[5, 6], 1])  # ElbowLeft -> WristLeft
    ax.plot3D(joints[[6, 7], 0], joints[[6, 7], 2], joints[[6, 7], 1])  # WristLeft -> HandLeft
    ax.plot3D(joints[[7, 21], 0], joints[[7, 21], 2], joints[[7, 21], 1])  # HandLeft -> HandTipLeft
    ax.plot3D(joints[[7, 22], 0], joints[[7, 22], 2], joints[[7, 22], 1])  # HandLeft -> ThumbLeft

    # Right arm
    ax.plot3D(joints[[2, 8], 0], joints[[2, 8], 2], joints[[2, 8], 1])  # Neck -> ShoulderRight
    ax.plot3D(joints[[8, 9], 0], joints[[8, 9], 2], joints[[8, 9], 1])  # ShoulderRight -> ElbowRight
    ax.plot3D(joints[[9, 10], 0], joints[[9, 10], 2], joints[[9, 10], 1])  # ElbowRight -> WristRight
    ax.plot3D(joints[[10, 11], 0], joints[[10, 11], 2], joints[[10, 11], 1])  # WristRight -> HandRight
    ax.plot3D(joints[[11, 23], 0], joints[[11, 23], 2], joints[[11, 23], 1])  # HandRight -> HandTipRight
    ax.plot3D(joints[[11, 24], 0], joints[[11, 24], 2], joints[[11, 24], 1])  # HandRight -> ThumbRight

    # Left leg
    ax.plot3D(joints[[0, 12], 0], joints[[0, 12], 2], joints[[0, 12], 1])  # SpineBase -> HipLeft
    ax.plot3D(joints[[12, 13], 0], joints[[12, 13], 2], joints[[12, 13], 1])  # HipLeft -> KneeLeft
    ax.plot3D(joints[[13, 14], 0], joints[[13, 14], 2], joints[[13, 14], 1])  # KneeLeft -> AnkleLeft
    ax.plot3D(joints[[14, 15], 0], joints[[14, 15], 2], joints[[14, 15], 1])  # AnkleLeft -> FootLeft

    # Right leg
    ax.plot3D(joints[[0, 16], 0], joints[[0, 16], 2], joints[[0, 16], 1])  # SpineBase -> HipRight
    ax.plot3D(joints[[16, 17], 0], joints[[16, 17], 2], joints[[16, 17], 1])  # HipRight -> KneeRight
    ax.plot3D(joints[[17, 18], 0], joints[[17, 18], 2], joints[[17, 18], 1])  # KneeRight -> AnkleRight
    ax.plot3D(joints[[18, 19], 0], joints[[18, 19], 2], joints[[18, 19], 1])  # AnkleRight -> FootRight

    # Spine to shoulder
    ax.plot3D(joints[[1, 20], 0], joints[[1, 20], 2], joints[[1, 20], 1])  # SpineMid -> SpineShoulder
    ax.plot3D(joints[[20, 2], 0], joints[[20, 2], 2], joints[[20, 2], 1])  # SpineShoulder -> Neck

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def show_upp(joints):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')
	draw_joints(joints, ax)
	set_axes_equal(ax)
	plt.show()