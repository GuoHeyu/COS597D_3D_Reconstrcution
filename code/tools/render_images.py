""" Copied from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Jiaming Sun and Yiming Xie. """

import pdb
import sys

sys.path.append('.')
import argparse
import json
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import pyrender
import torch
import trimesh
from tools.simple_loader import *

from tools.evaluation_utils import eval_depth, eval_mesh
from tools.visualize_metrics import visualize
import open3d as o3d
import ray
from PIL import Image


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="VisFusion ScanNet Testing")
    parser.add_argument('--n_images', default=3600, type=int)
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to dataset generated", default='example_data/ScanNet/scans_test')
    parser.add_argument("--gt_path", metavar="DIR",
                        help="path to ground truth ply file", default='example_data/ScanNet/gt_mesh')
    parser.add_argument("--camera_path", metavar="DIR",
                        default='example_data/ScanNet/intrinsic')
    parser.add_argument("--scene_input", required=True)
    parser.add_argument("--scene_output", required=True)


    # ray config
    parser.add_argument('--n_proc', type=int, default=1, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()

args = parse_args()


class Renderer():
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def process():
    save_path = os.path.join(args.data_path, args.scene_output)
    save_path_color = os.path.join(save_path, 'color')
    save_path_depth = os.path.join(save_path, 'depth')
    save_path_intrinsic = os.path.join(save_path, 'intrinsic')
    save_path_pose = os.path.join(save_path, 'pose')

    width, height = 640, 480
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(save_path_color):
        os.makedirs(save_path_color, exist_ok=True)    
    if not os.path.exists(save_path_depth):
        os.makedirs(save_path_depth, exist_ok=True)       
    if not os.path.exists(save_path_intrinsic):
        os.makedirs(save_path_intrinsic, exist_ok=True)    
    if not os.path.exists(save_path_pose):
        os.makedirs(save_path_pose, exist_ok=True)   
        

    n_imgs = args.n_images
    intrinsic_dir = os.path.join(args.camera_path, 'intrinsic_depth.txt')
    cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
    
    # Generate camera pose
    camera_pose = np.zeros((n_imgs, 4, 4))
    # x: 0~300, y: 0~150
    camera_pos_start = np.array([0,0,-2])
    camera_pos_end = np.array([5,5,-2])
    
#     camera_pos_start = np.reshape(np.array([0.838357,-0.248943,0.484960,3.932156,
# -0.544876, -0.355930, 0.759226, 5.240005,
# -0.016392, -0.900745, -0.434039, 1.384255,
# 0.000000, 0.000000, 0.000000, 1.000000]),(4,4))
#     camera_pos_end = camera_pos_start
    n_imgs_1d = np.int(np.sqrt(n_imgs))
    for i in range(n_imgs):
        camera_pose[i] = np.eye(4)
        camera_pose[i,0,3] = (camera_pos_end[0] - camera_pos_start[0]) * ((i % 60) / (n_imgs_1d-1)) + camera_pos_start[0]
        camera_pose[i,1,3] = (camera_pos_end[1] - camera_pos_start[1]) * ((i // 60) / (n_imgs_1d-1)) + camera_pos_start[1]
        camera_pose[i,2,3] = -2
        # camera_pose[i] = (camera_pos_end - camera_pos_start) * (i / n_imgs) + camera_pos_start

    mesh_file = os.path.join(args.gt_path, '%s.ply' % args.scene_input)
    mesh = trimesh.load(mesh_file, process=False)
    # mesh renderer
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)

    for i in range(n_imgs):
        print(args.scene_output, i, n_imgs)
        cam_pose = camera_pose[i]
        if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
            continue

        rgb_pred, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh_opengl)
        # print(np.max(depth_pred))
        
        np.savetxt(os.path.join(save_path_pose, str(i)+'.txt'), cam_pose, fmt='%.6f', delimiter=' ', header='', comments='')
        Image.fromarray(rgb_pred).save(os.path.join(save_path_color, str(i)+'.jpg'))
        Image.fromarray((depth_pred * 1000).astype(np.uint16)).save(os.path.join(save_path_depth, str(i)+'.png'))
        
    
    # Save intrinsic
    for root, _, files in os.walk(args.camera_path):
        for file in files:
            if file.endswith('.txt'):
                src_path = os.path.join(args.camera_path, file)
                dest_path = os.path.join(save_path_intrinsic, file)
                # Read and write the file
                with open(src_path, 'r') as src_file:
                    content = src_file.read()
                with open(dest_path, 'w') as dest_file:
                    dest_file.write(content)

def main():
    process()


if __name__ == "__main__":
    main()