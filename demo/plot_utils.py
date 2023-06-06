import numpy as np
import cv2 as cv

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from skeleton_utils import pose_connection, re_order_indices, re_order

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps, img):

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax, azim=0, elev=15):

    ax.view_init(azim=azim, elev=elev)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 320
    RADIUS_Z = 320

    #RADIUS = 0.72
    #RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)




def plot_skeleton_2d(ax, skeleton, with_index=True,
    title="title", xlabel="x", ylabel="y",
    invert_xaxis=False, invert_yaxis=True,
    xlim=None, ylim=None):

    for connect in pose_connection:

        point1_idx = connect[0]
        point2_idx = connect[1]

        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'black'

        plt.plot([int(point1[0]),int(point2[0])], 
                 [int(point1[1]),int(point2[1])], 
                 c=color, 
                 linewidth=2)

    if with_index:

        for i, joint in enumerate(skeleton):
            plt.text(joint[0], joint[1], str(i), color=color)

        plt.plot(skeleton[:,0], skeleton[:,1], 'ro', 2)       

    ax.set_aspect('equal')


    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if invert_xaxis:
        ax.invert_xaxis()

    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_skeleton_3d(ax, skeleton, elev=90, azim=90, roll=0, with_index=True,
    xlabel="x", ylabel="y", zlabel="z", title="title",
    invert_xaxis=True, invert_yaxis=False, invert_zaxis=True,
    xlim=None, ylim=None, zlim=None):

    # Set initial view angle
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # Plot bone connection
    for i in range(len(pose_connection)):

        start_index, end_index = pose_connection[i]
        start_point = skeleton[start_index]
        end_point = skeleton[end_index]
          
        x_arr = [start_point[0], end_point[0]]
        y_arr = [start_point[1], end_point[1]]
        z_arr = [start_point[2], end_point[2]]

        ax.plot(x_arr, y_arr, z_arr, linewidth=2, color='black')

    # plot joint points
    x_arr = []
    y_arr = []
    z_arr = []
    for i, joint in enumerate(skeleton):

        x = joint[0]
        y = joint[1]
        z = joint[2]

        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)

        if with_index: 
            ax.text(x, y, z, str(i), color='black')

    ax.scatter(x_arr, y_arr, z_arr, color="red")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if zlim is not None:
        ax.set_zlim(zlim)


    # Set axis attributes
    if invert_xaxis:
        ax.invert_xaxis()

    if invert_yaxis:
        ax.invert_yaxis()

    if invert_zaxis:
        ax.invert_zaxis()

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    ax.set_title(title)

    #ax.grid(True)
    

def figure_to_cv_image(fig):

    """Convert a Matplotlib figure to a numpy array"""
    fig.canvas.draw()
    
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    image = image[:, :, [2, 1, 0]]

    return image

def make_snapshot_plot(image, skel_2d, skel_3d, message=None):

    height = image.shape[1]
    width = image.shape[0]

    fig = plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(231)
    ax1.imshow(image)
    plot_skeleton_2d(ax1, skel_2d,
    title="Image", invert_xaxis=False, invert_yaxis=True,
    xlim=[0, width], ylim=[0, height])
    plt.text(20, 40, message, color='b')

    ax2 = plt.subplot(232)
    plot_skeleton_2d(ax2, skel_2d,
    title="2D key-point inputs",
    xlim=[0, width], ylim=[0, height])

    ax3 = plt.subplot(233, projection='3d')
    plot_skeleton_3d(ax3, skel_3d, elev=70, azim=80,
        invert_xaxis=True, invert_yaxis=False, invert_zaxis=False,
        xlabel="x", ylabel="y", zlabel="z", title="skel_3d prediction",
        xlim=[-width, width], ylim=[-height, height], zlim=[-width, width])

    skel_3d_xy = skel_3d[:, [0, 1]]
    ax4 = plt.subplot(234)
    plot_skeleton_2d(ax4, skel_3d_xy, title="skel_3d x-y", xlabel="x", ylabel="y",
        xlim=[-width, width], ylim=[-height, height])

    skel_3d_zy = skel_3d[:, [2, 1]]
    ax5 = plt.subplot(235)
    plot_skeleton_2d(ax5, skel_3d_zy, title="skel_3d z-y", xlabel="z", ylabel="y",
        xlim=[-width, width], ylim=[-height, height])

    skel_3d_xz = skel_3d[:, [0, 2]]
    ax6 = plt.subplot(236)
    plot_skeleton_2d(ax6, skel_3d_xz, title="skel_3d x-z", xlabel="x", ylabel="z",
        xlim=[-width, width], ylim=[-height, height])

    return fig


def draw_skeleton(ax, skeleton, gt=False, add_index=True):

    for segment_idx in range(len(pose_connection)):
        point1_idx = pose_connection[segment_idx][0]
        point2_idx = pose_connection[segment_idx][1]
        point1 = skeleton[point1_idx]
        point2 = skeleton[point2_idx]
        color = 'k' if gt else 'r'
        plt.plot([int(point1[0]),int(point2[0])], 
                 [int(point1[1]),int(point2[1])], 
                 c=color, 
                 linewidth=2)

    if add_index:
        for (idx, re_order_idx) in enumerate(re_order_indices):
            plt.text(skeleton[re_order_idx][0], skeleton[re_order_idx][1],
                str(idx+1), color='b')

    return
