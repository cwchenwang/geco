import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import kiui
from src.utils.train_util import instantiate_from_config
import copy

# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

class InstantMeshFT(nn.Module):
    def __init__(self, args, render_size=320):
        super().__init__()
        config = OmegaConf.load(args.config_path)
        config_name = os.path.basename(args.config_path).replace('.yaml', '')
        model_config = config.model_config
        infer_config = config.infer_config
        self.lrm_teacher = instantiate_from_config(model_config)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.render_size = render_size 
        self.input_size = 256

        if os.path.exists(infer_config.model_path):
            model_ckpt_path = infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.lrm_teacher.load_state_dict(state_dict, strict=True) 
        self.lrm_student = copy.deepcopy(self.lrm_teacher).train().requires_grad_(True)
        self.lrm_student.encoder.eval().requires_grad_(False)

        # TODO: be careful about the fov
        fov = 30.0 if args.zero123plus_v == 'v12' else 50.0
        self.lrm_teacher.init_flexicubes_geometry('cuda', fovy=fov)
        self.lrm_student.init_flexicubes_geometry('cuda', fovy=fov)
        self.lrm_teacher.eval().requires_grad_(False)
        # self.lrm_student.synthesizer.decoder.net_rgb.eval().requires_grad_(False)
    
    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lrm_teacher' in k:
                del state_dict[k]
        return state_dict
    
    def forward_lrm_generator(self, model, images, cameras, render_cameras, render_size=512):
        planes = torch.utils.checkpoint.checkpoint(
            model.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        out = model.forward_geometry(
            planes, 
            render_cameras, 
            render_size,
        )
        return planes, out
    
    def prepare_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['images_teacher'] # (B, 6, 3, 320, 320)
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)
        lrm_generator_input['images_teacher'] = images

        images = batch['images_student'] # (B, 6, 3, 320, 320)
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)
        lrm_generator_input['images_student'] = images

        # input cameras and render cameras
        input_c2ws = batch['input_c2ws'] # (B, 12, 4, 4)?
        input_Ks = batch['input_Ks']
        target_c2ws = batch['target_c2ws']

        render_c2ws = torch.cat([input_c2ws, target_c2ws], dim=1)
        render_w2cs = torch.linalg.inv(render_c2ws)

        input_extrinsics = input_c2ws.flatten(-2)
        input_extrinsics = input_extrinsics[:, :, :12]
        input_intrinsics = input_Ks.flatten(-2)
        input_intrinsics = torch.stack([
            input_intrinsics[:, :, 0], input_intrinsics[:, :, 4], 
            input_intrinsics[:, :, 2], input_intrinsics[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)
        # add noise to input_cameras
        # cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02

        lrm_generator_input['cameras'] = cameras
        lrm_generator_input['render_cameras'] = render_w2cs

        render_size = self.render_size

        lrm_generator_input['render_size'] = render_size

        return lrm_generator_input

    def forward(self, data, step_ratio=1.0):
        lrm_generator_input = self.prepare_batch_data(data)
        images_student = lrm_generator_input['images_student']
        images_teacher = lrm_generator_input['images_teacher']
        cameras = lrm_generator_input['cameras']
        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']

        planes = torch.utils.checkpoint.checkpoint(self.lrm_student.forward_planes, images_student, cameras, use_reentrant=False)
        render_out = self.lrm_student.forward_geometry(planes, render_cameras, render_size)

        if 'img' in data:
            loss, loss_dict = self.compute_loss_gt(render_out, data)
            results = {'images_gt': data['img'], 'images_pred': render_out['img'], 'images_student': images_student, 'images_teacher': images_teacher}
        else:
            with torch.no_grad():
                planes_teacher = self.lrm_teacher.forward_planes(images_teacher, cameras)
                render_out_teacher = self.lrm_teacher.forward_geometry(planes_teacher, render_cameras, render_size)
            loss, loss_dict = self.compute_loss(render_out, render_out_teacher)
            results = {'images_gt': render_out_teacher['img'], 'images_pred': render_out['img'], 'images_student': images_student, 'images_teacher': images_teacher}
        results['loss'] = loss
        # results['loss_triplane'] = loss_triplane
        # with torch.no_grad():
        #     results['psnr'] = -10 * torch.log10(torch.mean((results['images_gt'] - results['images_pred'])**2))
        results.update(loss_dict)
        return results
    
    def compute_loss_gt(self, render_out, render_gt):
        render_images = render_out['img']
        target_images = render_gt['img']
        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)

        render_alphas = render_out['mask']
        target_alphas = render_gt['mask']
        loss_mask = F.mse_loss(render_alphas, target_alphas)

        # render_depths = render_out['depth']
        # target_depths = render_gt['depth']
        # loss_depth = 0.5 * F.l1_loss(render_depths[target_alphas>0], target_depths[target_alphas>0])

        # render_normals = render_out['normal'] * 2.0 - 1.0
        # target_normals = render_gt['normal'] * 2.0 - 1.0
        # similarity = (render_normals * target_normals).sum(dim=-3).abs()
        # normal_mask = target_alphas.squeeze(-3)
        # loss_normal = 1 - similarity[normal_mask>0].mean()
        # loss_normal = 0.2 * loss_normal

        # flexicubes regularization loss
        sdf = render_out['sdf']
        sdf_reg_loss = render_out['sdf_reg_loss']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.lrm_student.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        loss = loss_mse + loss_lpips + loss_mask + loss_reg
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'loss_mse': loss_mse})
        loss_dict.update({f'loss_lpips': loss_lpips})
        loss_dict.update({f'loss_mask': loss_mask})
        loss_dict.update({f'loss_reg_sdf': sdf_reg_loss_entropy})
        loss_dict.update({f'loss_reg_surface': flexicubes_surface_reg})
        loss_dict.update({f'loss_reg_weights': flexicubes_weights_reg})
        # loss_dict.update({f'loss': loss})

        with torch.no_grad():
            psnr = -10 * torch.log10(loss_mse)
            loss_dict['psnr'] = psnr
        return loss, loss_dict

    def compute_loss(self, render_out, render_gt):
        render_images = render_out['img']
        target_images = render_gt['img']
        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)

        render_alphas = render_out['mask']
        target_alphas = render_gt['mask']
        loss_mask = F.mse_loss(render_alphas, target_alphas)

        render_depths = render_out['depth']
        target_depths = render_gt['depth']
        loss_depth = 0.5 * F.l1_loss(render_depths[target_alphas>0], target_depths[target_alphas>0])

        render_normals = render_out['normal'] * 2.0 - 1.0
        target_normals = render_gt['normal'] * 2.0 - 1.0
        similarity = (render_normals * target_normals).sum(dim=-3).abs()
        normal_mask = target_alphas.squeeze(-3)
        loss_normal = 1 - similarity[normal_mask>0].mean()
        loss_normal = 0.2 * loss_normal

        # flexicubes regularization loss
        sdf = render_out['sdf']
        sdf_reg_loss = render_out['sdf_reg_loss']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.lrm_student.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        loss = loss_mse + loss_lpips + loss_mask + loss_depth + loss_normal + loss_reg
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'loss_mse': loss_mse})
        loss_dict.update({f'loss_lpips': loss_lpips})
        loss_dict.update({f'loss_mask': loss_mask})
        loss_dict.update({f'loss_normal': loss_normal})
        loss_dict.update({f'loss_depth': loss_depth})
        loss_dict.update({f'loss_reg_sdf': sdf_reg_loss_entropy})
        loss_dict.update({f'loss_reg_surface': flexicubes_surface_reg})
        loss_dict.update({f'loss_reg_weights': flexicubes_weights_reg})
        # loss_dict.update({f'loss': loss})

        with torch.no_grad():
            psnr = -10 * torch.log10(loss_mse)
            loss_dict['psnr'] = psnr

        return loss, loss_dict