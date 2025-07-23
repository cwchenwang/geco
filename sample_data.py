from infer import Inferrer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from src.options import AllConfigs, Options
import os
import glob
import einops
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import json
import tqdm
import rembg
from kiui.cam import orbit_camera
from infer import default_rays_from_pose
from diffusers import DDIMScheduler

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

class ObjaverseDataset(Dataset):
    def __init__(self, path, source_size=256, low=0, high=10000):
        self.path = path 
        self.scenes = []
        subdirs = os.listdir(self.path)
        # subdirs = ['9000-9999']
        for subdir in subdirs:
            self.scenes += sorted(glob.glob(os.path.join(self.path, subdir, '*')))
        self.scenes = self.scenes[low:high]
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # return img and pose
        # filename = self.paths[index]
        filename = os.path.join(self.scenes[idx], '000.png')
        cond = np.array(Image.open( os.path.join(self.path, self.scenes[idx], '000.png') ))
        mask = cond[..., 3:4] / 255
        cond = cond[..., :3] * mask + (1 - mask) * 255
        # cond = np.array(Image.open( os.path.join(self.root_dir, self.scenes[idx], 'cond.png') ))
        return {
            'cond': cond.astype(np.uint8),
            'path': filename.split('/')[-2]
        }

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import random
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 42
seed_everything(seed)
g_cuda = torch.Generator(device='cuda')
g_cuda.manual_seed(42)

def generate(opt):
    bs = opt.sample_bs
    # distributed = torch.cuda.device_count() > 1
    # if distributed:
    #     torch.distributed.init_process_group(backend="nccl")
    #     gpu_id = int(os.environ["LOCAL_RANK"])
    #     if gpu_id == 0:
    #         print(f"Distributed session successfully initialized")
    # else:
    #     gpu_id = -1

    # if gpu_id == -1:
    #     device = torch.device(f"cuda")
    # else:
    #     device = torch.device(f"cuda:{gpu_id}")
    device = torch.device("cuda")

    # torch.cudnn.benchmark = True
    dataset = ObjaverseDataset(opt.sample_data_path, low=opt.sample_start, high=opt.sample_end)
    # dataset = ObjaverseLVISData('/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123/test_imgs')
    output_path = opt.sample_output_path
    # if gpu_id != -1:
    #     sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    # else:
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=bs, num_workers=0, pin_memory=True, shuffle=False,
    )

    # opt = tyro.cli(AllConfigs)
    inferrer = Inferrer(opt, device)

    # for LGM input, bs is num of scenes
    elevations, azimuths = [-30, 20, -30, 20, -30, 20], [30, 90, 150, 210, 270, 330]
    # rays_embeddings = inferrer.model.prepare_default_rays(device, elevations, azimuths).unsqueeze(0).repeat(bs, 1, 1, 1, 1)

    # B = zero123out.shape[0]
    with torch.no_grad():
        pipeline = inferrer.pipe
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.prepare()
        with tqdm.tqdm(loader) as pbar:
            for data in pbar:
                guidance_scale = 4.0
                prompt_embeds, cak = pipeline.prepare_conditions(data['cond'].to(device), guidance_scale=4.0)
                pipeline.scheduler.set_timesteps(50, device=device)
                timesteps = pipeline.scheduler.timesteps
                latents = torch.randn([bs, pipeline.unet.config.in_channels, 120, 80], device=device, dtype=torch.float16) 
                latents_init = latents.clone().detach() # initial noise z, to be saved

                with torch.no_grad():
                    for i, t in enumerate(timesteps):
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = pipeline.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cak,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if True:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        # compute the previous noisy sample x_t -> x_t-1
                        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    latents_out = unscale_latents(latents)
                    image = pipeline.vae.decode(latents_out / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image = unscale_image(image) # (B, 3, H, W)
                result = pipeline.image_processor.postprocess(image, output_type='pil')
                # result[0].save(f'./{data["path"][0]}.png')
                # images = np.array(images).permute(1, 2, 0).unsqueeze(0) # (B, 3, H, W)
                # images = images.cpu().numpy()
                input_image = einops.rearrange((image.clip(-1, 1) + 1) / 2, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2).reshape(-1, 3, 320, 320) # (B*V, 3, H, W)
                # imgs = []
                # for img in input_image:
                #     img_rm = rembg.remove((img.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) / 255.0
                #     img_rm  = img_rm[..., :3] * img_rm[..., 3:] + (1 - img_rm[..., 3:]) * 0.5
                #     imgs.append(img_rm.transpose(2, 0, 1))
                # input_image = torch.from_numpy(np.stack(imgs, axis=0)).to(device).float()
                image = einops.rearrange(input_image.reshape(-1, 6, 3, 320, 320), 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2)
                result = pipeline.image_processor.postprocess((image-0.5)*2, output_type='pil')
        
                for i, imgs in enumerate(image):
                    name = data["path"][i]
                    os.makedirs(os.path.join(output_path, name), exist_ok=True)
                    Image.fromarray(data['cond'][i].cpu().numpy().astype(np.uint8)).save(os.path.join(output_path, f'{name}/cond.png'))
                    np.save(os.path.join(output_path, f'{data["path"][i]}/z.npy'), latents_init[i].cpu().numpy())
                    np.save(os.path.join(output_path, f'{data["path"][i]}/latents_out.npy'), latents_out[i].cpu().numpy())
                    result[i].save(os.path.join(output_path, f'{name}/6view.png'))
                torch.cuda.empty_cache()


if __name__ == "__main__":

    opt = tyro.cli(AllConfigs)
    generate(opt)