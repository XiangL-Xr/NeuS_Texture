# !/usr/bin/python3
# coding: utf-8

import os, sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--match_type', type=str, default='exhaustive_matcher',
                    help='type of matcher used. Valid options: \
                    exhaustive_matcher sequential_matcher. Other matchers not supported at this time')
parser.add_argument('--scene_dir', type=str, default=None,
                    help='input scene directory')
parser.add_argument('--output_dir', type=str, default=None,
                    help='results output folder, include[image, mask, camera.npz]')

args = parser.parse_args()

if __name__ == '__main__':

    print('=' * 70)
    ## define input dir and output dir
    base_dir = args.scene_dir
    if args.output_dir is None:
        out_dir = os.path.join(base_dir, 'camera_npz')
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    poses_arr = np.load(os.path.join(base_dir, 'poses.npy')) # n_images, 15
    poses_hwf = poses_arr.reshape([-1, 3, 5])                # convert to n_images, 3, 5
    #print('pose_hwf', poses_hwf.shape)
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    #print('-- hwf: ', hwf)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)


    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        #cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        #cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)
    

    np.savez(os.path.join(out_dir, 'cameras.npz'), **cam_dict)
    print('=> Done with poses2init_npz!')
