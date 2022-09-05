# !/usr/bin/python3
# coding: utf-8

import os, sys
import numpy as np
import subprocess
import dataset.poses_utils.colmap_read_model as read_model

from glob import glob
from dataset.poses_utils.colmap_wrapper import run_colmap
from dataset.preprocess.generate_mask import get_mask 
from dataset.preprocess.preprocess_cameras import get_normalization


def load_colmap_data(real_dir):
    
    cameras_file = os.path.join(real_dir, 'sparse/0/cameras.bin')
    cam_data = read_model.read_cameras_binary(cameras_file)

    list_of_keys = list(cam_data.keys())
    cam = cam_data[list_of_keys[0]]
    print('=> Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h, w, f]).reshape([3, 1])

    images_file = os.path.join(real_dir, 'sparse/0/images.bin')
    img_data = read_model.read_images_binary(images_file)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [img_data[k].name for k in img_data]
    #print('--name: ', names)

    print('=> Images #', len(names))

    ## add by @lixiang
    perm_names = np.sort(names)           # Get a list of successfully reconstructed images.
    perm = np.argsort(names)
    #print('--perm_name:', perm_names)
    #print('--perm:', perm)
    for k in img_data:
        img = img_data[k]
        R = img.qvec2rotmat()
        t = img.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
        

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
    
    points3d_file = os.path.join(real_dir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3d_file)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], 
                            poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, perm_names

def save_poses(out_dir, poses, pts3d, perm):
    pts_arr = []
    #vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        #cams = [0] * poses.shape[-1]
        #for ind in pts3d[k].image_ids:
            #if len(cams) < ind - 1:
            #    print('=> ERROR: the correct camera poses for current points cannot be accessed')
            #    return
            #cams[ind - 1] = 1
        #vis_arr.append(cams)
    
    pts_arr = np.array(pts_arr)
    #vis_arr = np.array(vis_arr)
    #print('=> Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    print('=> Points', pts_arr.shape)

    #zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    #valid_z = zvals[vis_arr==1]
    #print('=> Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        #vis = vis_arr[:, i]
        #zs  = zvals[:, i]
        #zs  = zs[vis == 1]
        #close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        #print(i, close_depth, inf_depth)
        #save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
        save_arr.append(poses[..., i].ravel())
    save_arr = np.array(save_arr)

    np.save(os.path.join(out_dir, 'poses_bounds.npy'), save_arr)

def gen_poses(base_dir, out_dir, match_type):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(base_dir, 'sparse/0')):
        files_had = os.listdir(os.path.join(base_dir, 'sparse/0'))
    else:
        files_had = []
    
    if not all([f in files_had for f in files_needed]):
        print('=> Need to run COLMAP...')
        run_colmap(base_dir, match_type)
    else:
        print('=> Don\'t need to run COLMAP...')
    
    print('=> Post-colmap...')

    poses, pts3d, perm, perm_names = load_colmap_data(base_dir)
    save_poses(out_dir, poses, pts3d, perm)
    print('=> Done with imgs2poses!')

    return perm_names

def run_copy(in_name, out_name):
    strcmd = 'cp ' + in_name + ' ' + out_name
    subprocess.call(strcmd, shell=True)

def imgs2poses(scene_dir, match_type):
    
    if match_type != 'exhaustive_matcher' and match_type != 'sequential_matcher':
        print('=> ERROR: matcher type ' + match_type + ' is not valid. Aborting')
        sys.exit()

    #print('=' * 70)
    ## define input dir and output dir
    base_dir = scene_dir
    out_dir = os.path.join(base_dir, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    ## generate poses parameters
    perm_names = gen_poses(base_dir, out_dir, match_type)

    poses_arr = np.load(os.path.join(out_dir, 'poses_bounds.npy')) # n_images, 15
    poses_hwf = poses_arr.reshape([-1, 3, 5])                      # convert to n_images, 3, 5
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
        #print('-- first intrinsic:', intrinsic)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        #cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        #cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)
    
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)

    #image_list = glob(os.path.join(base_dir, 'images/*.*'))
    #image_list.sort()
    image_list = []
    masks_list = []
    for p_name in perm_names:
        #m_name = p_name.split('.')[0] + '.png'
        image_list.append(os.path.join(base_dir, 'images', p_name))
        if os.path.exists(os.path.join(base_dir, 'masks')):
            masks_list.append(os.path.join(base_dir, 'masks',  p_name))
    
    #print('--image_list: ', image_list)
    #print('--masks_list:', masks_list)

    mask_save_path = os.path.join(out_dir, 'masks')
    for i, image_path in enumerate(image_list):
        #img = cv.imread(image_path)
        #cv.imwrite(os.path.join(out_dir, 'image', '{:03d}.png'.format(i)), img, [int(cv.IMWRITE_PNG_COMPRESSION), 5])
        run_copy(image_path, os.path.join(out_dir, 'images', image_path.split('/')[-1]))
        
    if len(masks_list) > 0:
        for j, mask_path in enumerate(masks_list):
            #mask = cv.imread(mask_path)
            #cv.imwrite(os.path.join(out_dir, 'mask', '{:03d}.png'.format(j)), mask)
            run_copy(mask_path, os.path.join(out_dir, 'masks', mask_path.split('/')[-1]))
    else:
        # generate mask add by @lixiang
        try:
            print("=> start generate masks...")
            get_mask(image_list, mask_save_path)
        except:
            print("=> generate mask error!")

    np.savez(os.path.join(out_dir, 'cameras_init.npz'), **cam_dict)
    print('=> Done with poses2init_npz!')

    # generate cameras.npz, the file include world_matrix and scale_matrix
    get_normalization(out_dir)         # this step will generate cameras.npz in out_dir/
    print('=> Done with poses2npz!')