import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from src.options import Options
# from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import json
from PIL import Image
import einops
from src.utils.camera_util import spherical_camera_pose, FOV_to_intrinsics
# from core.dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class NoiseImageDataset(Dataset):
    def __init__(self, opt: Options, training=True):
        self.path = opt.data_path
        self.scenes = sorted(os.listdir(self.path))
        import warnings
        warnings.warn('not using full dataset yet')
        self.opt = opt
        self.training = training

        self.gt_scenes = {}
        self.gt_path = '/mnt/lingjie_cache/lvis_dataset/testing'
        subdirs = os.listdir(self.gt_path)
        # subdirs = ['9000-9999']
        for subdir in subdirs:
            # self.gt_scenes += sorted(glob.glob(os.path.join(self.gt_path, subdir, '*')))
            if os.path.isdir(os.path.join(self.gt_path, subdir)):
                for sub in os.listdir(os.path.join(self.gt_path, subdir)):
                    self.gt_scenes[sub] = os.path.join(self.gt_path, subdir, sub)
        self.scenes = [scene for scene in self.scenes if scene in self.gt_scenes]

        if self.training:
            self.scenes = self.scenes[:-opt.batch_size]
        else:
            self.scenes = self.scenes[-opt.batch_size:]
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        results = {}
        vids = np.arange(6).tolist() + np.random.permutation(np.arange(6, 50)).tolist()

        z = torch.from_numpy(np.load(os.path.join(self.path, self.scenes[idx], 'z.npy')))
        latents = torch.from_numpy(np.load(os.path.join(self.path, self.scenes[idx], 'latents_out.npy')))
        cond = np.array(Image.open( os.path.join(self.path, self.scenes[idx], 'cond.png') ))
        cond_img = np.array(Image.open( os.path.join(self.path, self.scenes[idx], 'cond.png') ).resize((320, 320), Image.LANCZOS)).transpose(2, 0, 1) / 255.0

        six_view_path = os.path.join(self.path, self.scenes[idx], '6view.png')
        if os.path.exists(six_view_path):
            six_view = np.array(Image.open(six_view_path)) / 255.0
            # results['images_teacher'] = torch.from_numpy(np.array(Image.open(six_view_path))).permute(2, 0, 1).float() / 255.0
            six_view = einops.rearrange(six_view, '(h2 h) (w2 w) c -> (h2 w2) c h w', h2=3, w2=2).astype(np.float32) # (6, 3, 320, 320)
            if self.opt.use_cond:
                six_view[0] = cond_img
            images_input = torch.from_numpy(six_view)
            results['images_teacher'] = images_input

        cam_poses = []
        elevations = np.random.randint(-90, 90, size=self.opt.num_views)
        azimuths = np.random.randint(0, 360, size=self.opt.num_views)
        if self.opt.zero123plus_v == 'v11':
            elevations[:6] = [30, -20, 30, -20, 30, -20]
            azimuths[:6] = [30, 90, 150, 210, 270, 330]
        else:
            raise NotImplementedError
        
        if self.opt.use_cond:
            elevations[0] = 0
            azimuths[0] = 0
        
        # for i, vid in enumerate(vids[:self.opt.num_views]):

        #     from kiui.cam import orbit_camera
        #     c2w = orbit_camera(elevations[i], azimuths[i], radius=1.5)
        #     c2w = torch.from_numpy(c2w)
        #     c2w[:3, 3] *= self.opt.cam_radius / 1.5 # TODO: caution camera radius

        #     cam_poses.append(c2w)
        c2ws = spherical_camera_pose(azimuths, elevations, self.opt.cam_radius)
        c2ws = c2ws.float()
        results['input_c2ws'] = c2ws[:6]
        results['target_c2ws'] = c2ws[6:]

        Ks = FOV_to_intrinsics(self.opt.fovy).unsqueeze(0).repeat(elevations.shape[0], 1, 1).float()
        results['input_Ks'] = Ks[:6]
        
        # cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # images_input = F.interpolate(images_input, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # cam_poses_input = cam_poses[:self.opt.num_input_views].clone()
        # images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        assert os.path.exists(os.path.join(self.path, self.scenes[idx], 'onestep-20k.png')), f'{os.path.join(self.path, self.scenes[idx], "onestep-20k.png")} does not exist'
        onestep_path = os.path.join(self.path, self.scenes[idx], 'onestep-20k.png')
        onestep_view = np.array(Image.open(onestep_path)) / 255.0
        onestep_view = einops.rearrange(onestep_view, '(h2 h) (w2 w) c -> (h2 w2) c h w', h2=3, w2=2).astype(np.float32) # (6, 320, 320, 3)
        if self.opt.use_cond:
            onestep_view[0] = cond_img
        onestep_view = torch.from_numpy(onestep_view)
        results['images_student'] = onestep_view
        # onestep_view = F.interpolate(onestep_view, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # onestep_view = TF.normalize(onestep_view, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        results['cond'] = cond
        results['z'] = z
        results['index'] = self.scenes[idx]
        # if not images_teacher in results:
        #     results['images_teacher'] = onestep_view

        # # Load GT images
        # cond_cam = np.load(os.path.join(self.gt_scenes[self.scenes[idx]], '000.npy'), allow_pickle=True).item()
        # elevations, azimuths = [], []
        # images, masks = [], []
        # vids = np.arange(0, 7).tolist() + np.random.permutation(np.arange(7, 50)).tolist()[:self.opt.num_views]
        # for vid in vids:
        #     image_path = os.path.join(self.gt_scenes[self.scenes[idx]], f'{vid:03d}.png')
        #     camera_path = os.path.join(self.gt_scenes[self.scenes[idx]], f'{vid:03d}.npy')

        #     cam = np.load(camera_path, allow_pickle=True).item()
        #     azimuth = cam['azimuth'] - cond_cam['azimuth']
        #     if azimuth < 0:
        #         azimuth += 360
        #     if azimuth > 360:
        #         azimuth -= 360
        #     azimuths.append(azimuth)
        #     elevations.append(cam['elevation'])
    
        #     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     image = image.astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
        #     image = torch.from_numpy(image)
        #     image = image.permute(2, 0, 1) # [4, 512, 512]
        #     mask = image[3:4] # [1, 512, 512]
        #     image = image[:3] * mask + (1 - mask) * self.opt.bg # [3, 512, 512], to white bg
        #     image = image[[2,1,0]].contiguous() # bgr to rgb
        #     images.append(image)
        #     masks.append(mask.squeeze(0))

        # c2ws = spherical_camera_pose(azimuths[7:], elevations[7:], self.opt.cam_radius)
        # c2ws = c2ws.float()
        # results['cond'] = images[0] * 255.0
        # results['input'] = torch.stack(images[1:7]) # (6, 320, 320, 3)
        # results['mask'] = torch.stack(masks[1:], dim=0)
        # results['img'] = torch.stack(images[1:], dim=0)
        # results['target_c2ws'] = c2ws
        # Ks = FOV_to_intrinsics(self.opt.fovy).unsqueeze(0).repeat(len(azimuths), 1, 1).float()
        # results['input_Ks'] = Ks

        return results

class ObjaverseRenderingDataset(Dataset):
    def __init__(self, opt: Options, training=True):

        self.path = opt.data_path
        self.scenes = []
        subdirs = os.listdir(self.path)
        # subdirs = ['9000-9999']
        for subdir in subdirs:
            self.scenes += sorted(glob.glob(os.path.join(self.path, subdir, '*')))
        self.opt = opt
        self.training = training
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        results = {}
        vids = np.arange(1, 7).tolist() + np.random.permutation(np.arange(7, 50)).tolist()[self.opt.num_views]

        cond = np.array(Image.open( os.path.join(self.path, self.scenes[idx], '000.png') ))
        mask = cond[..., 3:4] / 255
        cond = cond[..., :3] * mask + (1 - mask) * int(self.opt.bg * 255)
        cond = cond.astype(np.uint8)

        cond_cam = np.load(os.path.join(self.path, self.scenes[idx], '000.npy'), allow_pickle=True).item()
        elevations, azimuths = [], []
        images, masks, cam_poses = [], [], []
        for vid in vids:
            image_path = os.path.join(self.scenes[idx], f'{vid:03d}.png')
            camera_path = os.path.join(self.scenes[idx], f'{vid:03d}.npy')

            cam = np.load(camera_path, allow_pickle=True).item()
            from kiui.cam import orbit_camera
            azimuth = cam['azimuth'] - cond_cam['azimuth']
            if azimuth < 0:
                azimuth += 360
            if azimuth > 360:
                azimuth -= 360
            azimuths.append(azimuth)
            elevations.append(cam['elevation'])
    
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (320, 320))
            image = image.astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) * self.opt.bg # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
        
        print(elevations, azimuths)
        c2ws = spherical_camera_pose(azimuths, elevations, self.opt.cam_radius)
        c2ws = c2ws.float()
        results['input_c2ws'] = c2ws[:6]
        results['target_c2ws'] = c2ws[6:]

        Ks = FOV_to_intrinsics(self.opt.fovy).unsqueeze(0).repeat(elevations.shape[0], 1, 1).float()
        results['input_Ks'] = Ks[:6]
        results['input_images'] = images[:6]
        results['target_images'] = images[6:]

        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]

        results['cond'] = cond
        results['index'] = self.scenes[idx]

        return results