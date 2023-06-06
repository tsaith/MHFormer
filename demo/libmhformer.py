import glob
import os
import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d


def interpolate_array_1d(a, dims_out):

    dims_in = a.shape[0]
    f = interp1d(np.linspace(0, 1, dims_in), a, kind='linear')
    dims_out = np.linspace(0, 1, dims_out)
    out = f(dims_out)

    return out

def interpolate_keypoints_2d(keypoints, num_frames_out=81):

    num_joints = keypoints.shape[1]
    num_dims = keypoints.shape[2]

    dtype = keypoints.dtype
    keypoints_out = np.zeros([num_frames_out, num_joints, num_dims], dtype=dtype)

    for j in range(num_joints):
        for d in range(num_dims):

            kp_hist_in = keypoints[:, j, d].copy()
            kp_hist_out = interpolate_array_1d(kp_hist_in, num_frames_out)
            keypoints_out[:, j, d] = kp_hist_out

    return keypoints_out


def rescale_skeleton_2d(skel_in, frame_width, frame_height):

    num_joints = skel_in.shape[0]

    skel = skel_in.copy()

    # Rescale
    x_min = np.min(skel[:, 0])
    x_max = np.max(skel[:, 0])

    y_min = np.min(skel[:, 1])
    y_max = np.max(skel[:, 1])

    human_width = abs(x_max - x_min)
    human_height = abs(y_max - y_min)

    resize_factor = 0.8
    err_tol = 1.0e-8
    if human_width > err_tol and human_height > err_tol:

        x_ratio = frame_width/human_width
        y_ratio = frame_height/human_height

        target_ratio = resize_factor*min(x_ratio, y_ratio)

        for i in range(num_joints):
            skel[i, :] = skel[i, :]*target_ratio

    # Set the pelvis as center
    x0 = 0.5*frame_width
    y0 = 0.5*frame_height

    pelvis = skel[0]
    shift_dx = x0 - pelvis[0]
    shift_dy = y0 - pelvis[1]

    for i in range(num_joints):
        skel[i, 0] = skel[i, 0] + shift_dx
        skel[i, 1] = skel[i, 1] + shift_dy

    return skel


def rescale_skeleton_3d(skel_3d, skel_2d):

    skel_out = skel_3d.copy()

    xmin_2d = np.min(skel_2d[:, 0])
    xmax_2d = np.max(skel_2d[:, 0])

    ymin_2d = np.min(skel_2d[:, 1])
    ymax_2d = np.max(skel_2d[:, 1])

    xmin_3d = np.min(skel_3d[:, 0])
    xmax_3d = np.max(skel_3d[:, 0])

    ymin_3d = np.min(skel_3d[:, 1])
    ymax_3d = np.max(skel_3d[:, 1])

    dx_2d = xmax_2d - xmin_2d 
    dy_2d = ymax_2d - ymin_2d 

    dx_3d = xmax_3d - xmin_3d
    dy_3d = ymax_3d - ymin_3d

    ratio_x = dx_2d/dx_3d
    ratio_y = dy_2d/dy_3d

    skel_out[:, 0] = skel_3d[:, 0]*ratio_x
    skel_out[:, 1] = skel_3d[:, 1]*ratio_y
    skel_out[:, 2] = skel_3d[:, 2]*ratio_x

    return skel_out

def update_skeleton_depth_only(skel_3d_in, skel_2d_in):

    skel_2d = skel_2d_in.copy()
    skel_3d = skel_3d_in.copy()

    pelvis_2d = skel_2d[0]
    skel_2d = skel_2d - pelvis_2d

    skel_3d[:, 0] = skel_2d[:, 0]
    skel_3d[:, 1] = skel_2d[:, 1]

    return skel_3d

def get_hip_width_2d(skel_2d):

    left_hip = skel_2d[4]
    right_hip = skel_2d[1]

    width = np.linalg.norm(right_hip-left_hip)

    return width

def get_hip_dist_2d(skel_2d):

    left_hip = skel_2d[4]
    right_hip = skel_2d[1]

    diff = right_hip-left_hip

    dx = abs(diff[0])
    dy = abs(diff[1])

    return dx, dy


def img2video(video_path, output_dir):
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS)) + 5

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv.imread(names[0])
    size = (img.shape[1], img.shape[0])

    video_filename = os.path.basename(video_path)
    videoWrite = cv.VideoWriter(output_dir + video_filename, fourcc, fps, size) 

    for name in names:
        img = cv.imread(name)
        videoWrite.write(img)

    videoWrite.release()
