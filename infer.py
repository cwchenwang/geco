import os
import tyro
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import random
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import pickle
import einops
import time
import kiui
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler, LCMScheduler
from src.options import AllConfigs
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import FOV_to_intrinsics, get_zero123plus_input_cameras, get_circular_camera_poses, get_zero123plus_input_cameras_v11
from src.utils.mesh_util import save_obj, save_obj_with_mtl, save_glb
from src.utils.infer_util import remove_background, resize_foreground, save_video

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, 
    guidance_scale=1.0, cross_attention_kwargs={}, scheduler=None, lora_v=False, model='sd'):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if type(t) == int:
        t = torch.tensor([t] * batch_size, device=noisy_latents.device)
    # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    if guidance_scale == 1.:
        cak = {}
        if cross_attention_kwargs is not None:
            for key in cross_attention_kwargs:
                if isinstance(cross_attention_kwargs[key], torch.Tensor):
                    cak[key] = cross_attention_kwargs[key][batch_size:]
                else:
                    cak[key] = cross_attention_kwargs[key]
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cak).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
        if model == 'unet_if':
            noise_pred, _ = noise_pred.split(3, dim=1)
    else:
        t = torch.cat([t] * 2)
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if model == 'unet_if':
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def get_alpha(scheduler, t, device):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    return alpha_t, sigma_t

def predict_x0(unet, noisy_latents, text_embeddings, t, 
        guidance_scale=1.0, cross_attention_kwargs={}, 
        scheduler=None, lora_v=False, model='sd'):
    alpha_t, sigma_t = get_alpha(scheduler, t, noisy_latents.device)
    noise_pred = predict_noise0_diffuser(
        unet, noisy_latents, text_embeddings, t=t,
        guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, 
        scheduler=scheduler, model=model
    )
    return (noisy_latents - noise_pred * sigma_t) / alpha_t

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def encode_image(image, vae, is_zero123plus=True):
    if is_zero123plus:
        image = scale_image(image)
        image = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        image = scale_latents(image)
    else:
        image = vae.encode(image, return_dict=False)[0] * vae.config.scaling_factor
    return image

def decode_latents(latents, decoder, is_zero123plus=True):
    if is_zero123plus:
        latents = unscale_latents(latents)
        latents = latents / decoder.config.scaling_factor
        image = decoder.decode(latents, return_dict=False)[0]
        # image = decoder(latents)
        image = unscale_image(image)
    else:
        image = decoder.decode(latents, return_dict=False)[0]
    return image

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames

class Inferrer:
    def __init__(self, opt, device='cuda'):
        self.opt = opt        
        self.device = torch.device(device)

        self.load_reconstruction_model()
        self.model.eval()

        self.pipe = self.load_pipeline(opt.pipeline)
    
    def load_pipeline(self, pipeline):
        import sys
        sys.path.append('./pipelines/')
        pipe_path = 'sudo-ai/zero123plus-v1.1' if self.opt.zero123plus_v == 'v11' else 'sudo-ai/zero123plus-v1.2'
        pipe = DiffusionPipeline.from_pretrained(
            pipe_path,
            custom_pipeline="./pipelines/zero123plus.py",
            torch_dtype=torch.float16,
        ).to(self.device)
        if self.opt.zero123plus_v == 'v11':
            unet_ckpt_path = './ckpts/diffusion_pytorch_model_v11.bin'
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict, strict=True)
        pipe.prepare()
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
        if pipeline == 'zero123plus1step':
            pipe.prepare()
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            resume_data = pickle.load(open(self.opt.resume_pkl, 'rb'))
            copy_params_and_buffers(resume_data['G'], pipe.unet, require_all=False)
            pipe.unet.eval()
            pipe.unet.is_generator = True
        return pipe
    
    def load_reconstruction_model(self):
        print('Loading reconstruction model ...')
        self.config = OmegaConf.load(self.opt.config_path)
        config_name = os.path.basename(self.opt.config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        self.model = instantiate_from_config(self.model_config)
        if self.opt.resume is not None:
            if self.opt.resume.endswith('safetensors'):
                ckpt = load_file(self.opt.resume, device='cpu')
            else:
                raise NotImplementedError
            ckpt = {k[12:]: v for k, v in ckpt.items() if k.startswith('lrm_student.')}
            self.model.load_state_dict(ckpt, strict=True)
            print(f'[INFO] Loaded checkpoint from {self.opt.resume}')
        else:
            if os.path.exists(self.infer_config.model_path):
                model_ckpt_path = self.infer_config.model_path
            else:
                model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
            print(model_ckpt_path)
            state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
            state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
            self.model.load_state_dict(state_dict, strict=True)
            print(f'loading from initial reconstruction model') 

        self.model = self.model.to(self.device)
        IS_FLEXICUBES = True
        if IS_FLEXICUBES:
            self.model.init_flexicubes_geometry(self.device, fovy=self.opt.fovy)
        self.model = self.model.eval()

    def generate_mv_image(self, image, seed=42):
        t1 = time.time()
        if self.opt.pipeline == 'zero123plus':
            mv_image = self.pipe(Image.fromarray(image.astype(np.uint8)), num_inference_steps=1, guidance_scale=1.0).images[0]
        else:
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(image.astype(np.uint8), guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs
            # print(f'preparing time: {time.time() - t1:.2f}s')
            with torch.no_grad():
                gen = self.pipe.unet if self.opt.model_type == 'LGM' else self.model.gen
                out = predict_x0(gen, torch.randn([1, 4, 120, 80], dtype=text_embeddings.dtype, device=self.device, generator=torch.Generator(device=self.device).manual_seed(seed)), text_embeddings, t=self.opt.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
                out = (decode_latents(out, self.pipe.vae, True)[0] + 1)*127.5 # (-1, 1) -> (0, 255)
            mv_image = Image.fromarray(out.float().permute(1, 2, 0).detach().clip(0, 255).cpu().numpy().astype(np.uint8))
        mv_image = np.array(mv_image)
        mv_image = einops.rearrange(mv_image, '(h2 h) (w2 w) c -> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) # [6, 320, 320, 3]
        mv_image = mv_image.astype(np.float32) / 255.0
        print(f'generation time: {time.time() - t1:.2f}s')
        return mv_image
    
    @torch.no_grad()
    def onestep_gen(self, z, cond):
        text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
        cross_attention_kwargs_stu = cross_attention_kwargs
        out = predict_x0(self.pipe.unet, z, text_embeddings, t=self.opt.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        out = (decode_latents(out, self.pipe.vae, True) + 1)*127.5
        return out

    def infer(self, cond, out_dir, use_cond=None, elevation=0.0, seed=42, demo=False):
        use_cond = use_cond if use_cond is not None else self.opt.use_cond
        os.makedirs(out_dir, exist_ok=True)
        t1 = time.time()
        if os.path.exists(os.path.join(out_dir, 'mv_image.png')):
            print('existing...')
            mv_image = np.array(Image.open(os.path.join(out_dir, 'mv_image.png'))) / 255.0
            mv_image = einops.rearrange(mv_image, 'h (h2 w2 w) c -> (h2 w2) h w c', h2=3, w2=2) # [6, 320, 320, 3]
        else:
            mv_image = self.generate_mv_image(cond, seed=seed)
            if use_cond:
                mv_image[0] = np.array(Image.fromarray(cond.astype(np.uint8)).resize((320, 320))) / 255.0
            kiui.write_image(os.path.join(out_dir, 'mv_image.png'), mv_image.transpose(1, 0, 2, 3).reshape(-1, mv_image.shape[1]*mv_image.shape[0], 3))
        input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(self.device) # [6, 3, 256, 256]
        if self.opt.zero123plus_v == 'v11':
            print('using zero123plus v1.1')
            input_cameras = get_zero123plus_input_cameras_v11(batch_size=1, radius=self.opt.distance, fov=50, input_elevation=elevation, use_cond=use_cond).to(self.device) 
        else:
            print('using zero123plus v1.2')
            input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)
        IS_FLEXICUBES = True
        chunk_size = 20 if IS_FLEXICUBES else 1
        with torch.no_grad():
            planes = self.model.forward_planes(input_image.unsqueeze(0), input_cameras)
            mesh_path_idx = os.path.join(out_dir, f'mesh.obj')

            mesh_out = self.model.extract_mesh(planes, use_texture_map=self.opt.export_texmap, **self.infer_config)
            print('Mesh generation time', time.time() - t1)
            vertices, faces, vertex_colors = mesh_out
            if demo: vertices = vertices[:, [1, 2, 0]]
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
            print(f"Mesh saved to {mesh_path_idx}")
            if demo: return

            render_cameras = get_render_cameras(batch_size=1, M=16, radius=self.opt.distance, elevation=0.0, is_flexicubes=IS_FLEXICUBES).to(self.device)
            
            render_size = self.infer_config.render_resolution
            pred_images = render_frames(self.model, planes, render_cameras=render_cameras, render_size=render_size, chunk_size=chunk_size, is_flexicubes=IS_FLEXICUBES) # torch.Size([16, 3, 512, 512]), 0~1
            pred_images = (pred_images.permute(0,2,3,1).detach().cpu().numpy() * 255).astype(np.uint8)
            os.makedirs(os.path.join(out_dir, 'imgs'), exist_ok=True)
            for (i, img) in enumerate(pred_images):
                kiui.write_image(f'{out_dir}/imgs/{i:03d}.png', img)

            video_path_idx = os.path.join(out_dir, f'video.mp4')
            render_size = self.infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=self.opt.distance, 
                elevation=0.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(self.device)
            
            frames = render_frames(self.model,  planes, render_cameras=render_cameras, render_size=render_size, chunk_size=120, is_flexicubes=IS_FLEXICUBES)
            save_video(frames, video_path_idx, fps=30)
            print(f"Video saved to {video_path_idx}")
            print(f"Total time: {time.time() - t1:.2f}s")   

def pad_image(image, ratio=0.80):
    h, w = image.shape[:2]
    max_size = max(h, w)
    side_len = int(max_size / ratio)
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image
    return Image.fromarray(padded_image)

if __name__ == "__main__":
    opt = tyro.cli(AllConfigs)
    inferrer = Inferrer(opt)
    seed_everything(42)

    import glob
    
    file_paths = {opt.test_path.split('/')[-1][:-4]: opt.test_path} 

    # file_paths = {path.split('/')[-1][:-4]: path for path in paths}
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    seed = 42
    # testing
    for i, key in enumerate(file_paths.keys()):
            # if os.path.exists(os.path.join(output_dir, key)):
            #     print(f'{key} already exists, skipping')
            #     continue
        for seed in [42]:
            cond = Image.open(f'{file_paths[key]}')
            if cond.mode != 'RGBA':
                print('removing background, which may take extra time')
                cond = rembg.remove(cond)
            if cond.size[0] != cond.size[1]:
                cond = pad_image(np.array(cond))
            cond = np.array(cond.resize((512, 512)))

            mask = cond[..., 3:4] / 255
            cond_input = cond[..., :3] * mask + (1 - mask) * 255
            cond = cond[..., :3] * mask + (1 - mask) * 255

            inferrer.infer(cond, f'{output_dir}/{key}', elevation=opt.input_elevation, seed=seed)