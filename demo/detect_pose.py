import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy

sys.path.append(os.getcwd())
from model.mhformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from skyeye.pose_estimation.mhformer import interpolate_keypoints_2d
from skyeye.pose_estimation.mhformer import rescale_skeleton_2d
from skyeye.pose_estimation.mhformer import show2Dpose, show3Dpose
from skyeye.pose_estimation.mhformer import img2video

from skyeye.pose_estimation.utils import plot_skeleton_2d

from skyeye.utils import Timer


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successfully!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir, debug=False):

    use_gpu = True
    #num_frames_model = 351
    num_frames_model = 81
    #num_frames_model = 27

    num_frames_use = 5

    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, num_frames_model
    args.previous_dir = f'checkpoint/pretrained/{num_frames_model}'
    #args.previous_dir = 'checkpoint/pretrained/351'

    args.pad = (args.frames - 1) // 2
    args.n_joints, args.out_joints = 17, 17

    timer = Timer()

    ## Reload
    if use_gpu:
        model = Model(args).cuda()
    else:
        model = Model(args)

    model_dict = model.state_dict()
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"frame_width, frame_height: {frame_width}, {frame_height}")   

    output_dir_2D_rescale = os.path.join(output_dir, "pose2D_rescale")
    os.makedirs(output_dir_2D_rescale, exist_ok=True)

    # Rescale keypoints
    keypoints_rescale = keypoints.copy()
    for i in tqdm(range(video_length)):
        keypoints_rescale[0, i, :, :] = rescale_skeleton_2d(keypoints_rescale[0, i, :, :],
            frame_width, frame_height)

        if debug:
            fig = plt.figure(figsize=(9.6, 5.4))
            ax = plt.subplot(111)
            
            plot_skeleton_2d(ax, keypoints_rescale[0, i, :, :], with_index=True,
                title="2D rescale", xlabel="x", ylabel="y",
                invert_xaxis=False, invert_yaxis=True,
                xlim=[0, frame_width], ylim=[0, frame_height])
    
            filepath = f"{output_dir_2D_rescale}/{i:04d}_2D_rescale.png"
            plt.savefig(filepath, dpi=200, format='png', bbox_inches = 'tight')
            plt.close(fig)

    #keypoints = keypoints_rescale

    ## 3D
    print('\nGenerating 3D pose...')
    output_3d_all = []
    for i in tqdm(range(video_length)):

        ret, img = cap.read()
        img_size = img.shape
       
        input_2D_no = keypoints_rescale[0][i]
        input_2D_no = np.tile(input_2D_no, (1, 1, 1))

        num_joints = 17
        num_dims = 2
        input_2D_no = np.zeros([num_frames_use, num_joints, num_dims], dtype=np.float32)

        for ii in range(num_frames_use):
            j = i - ii
            k = num_frames_use - 1 - ii 
            input_2D_no[k] = keypoints[0][j]

        input_2D_no = interpolate_keypoints_2d(input_2D_no, num_frames_out=num_frames_model) 

        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        if use_gpu:
            input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        else:
            input_2D = torch.from_numpy(input_2D.astype('float32')).cpu()

        N = input_2D.size(0)

        ## estimation
        timer.tic()

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])
        
        dt = timer.toc()
        fps = 1.0/dt

        print(f"fps: {fps}")

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, args.pad].unsqueeze(1) 
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        output_3d_all.append(post_out)

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        input_2D_no = input_2D_no[args.pad]

        ## 2D
        image = show2Dpose(keypoints[0, i, :, :], copy.deepcopy(img))
        #image = show2Dpose(input_2D_no, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)

        filepath = f"{output_dir_2D}/{i:04d}_2D.png"
        cv2.imwrite(filepath, image)

        ## 3D
        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(post_out, ax)

        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)

        filepath = f"{output_dir_3D}/{i:04d}_3D.png"
        plt.savefig(filepath, dpi=200, format='png', bbox_inches = 'tight')
        plt.close(fig)
        
    ## save 3D keypoints
    output_3d_all = np.stack(output_3d_all, axis = 0)
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('Generating 3D pose successfully!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 102
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    if os.path.exists(video_path):
        print(f"{video_path} is found.")
    else:
        print(f"Error: {video_path} doesn't exist!")

    #get_pose2D(video_path, output_dir)
    get_pose3D(video_path, output_dir)
    img2video(video_path, output_dir)
    print('Generating demo successful!')


