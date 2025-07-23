import os
import time
import copy
import json
import pickle
import psutil
from PIL import Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils import distributed as dist

import torch.nn.functional as F
from tqdm import tqdm
import sys

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

from utils.diff import extract_lora_diffusers, predict_noise0_diffuser, predict_x0, predict_x0_ldm
from utils.saving import *
from training.random_camera import *

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_list):
        self.prompt_list = prompt_list

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        sample = self.prompt_list[idx]

        return sample

def decode_latents(latents, vae, is_zero123plus):
    if is_zero123plus:
        latents = unscale_latents(latents)
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = unscale_image(image)
    else:
        image = vae.decode(latents, return_dict=False)[0]
    return image
    
#---------------------------------------------------------------------------
def training_loop(
    run_dir                 = '.',      # Output directory.
    dataset_kwargs          = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},    
    Diff_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},
    Diff_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_diff_kwargs        = {},
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    augment_p               = 0,        # Initial value of augmentation probability.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    state_dump_ticks        = 50,
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_state_dump       = None,
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    network_pkl             = None,
    curr_time               = '00',
    gen_type                = '3d',
    arch                    = 'sd',     
    resolution              = 64,
    ratio                   = 1,
    label_dim               = 0,
    num_channels            = 3,
    grad_clip               = None,
    output_clip             = False,
    loss_type               = 'sds',
    n_images                = 1,
    max_iters               = 10000,
    **useless_kwargs
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.

    # Load dataset
    dist.print0('Loading dataset...')
    data_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(dataset=data_obj, rank=rank, num_replicas=num_gpus, seed=random_seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=data_obj, sampler=dataset_sampler, batch_size=batch_size//num_gpus))

    # Construct networks.
    dist.print0('Constructing networks...')

    sys.path.append('./training')
    dtype = torch.float16 if Diff_kwargs.dtype == 'fp16' else torch.float32
    pipe = DiffusionPipeline.from_pretrained(Diff_kwargs.model_path, custom_pipeline=Diff_kwargs.custom_pipeline, torch_dtype=dtype).to(device)
    cross_attention_kwargs = {}
    cross_attention_kwargs_stu = {}
    if Diff_kwargs.cond == 't2i':
        if Diff_kwargs.prompt_path is not None:
            if Diff_kwargs.prompt_path.endswith('.json'):
                prompts = json.load(open(Diff_kwargs.prompt_path, 'r'))['dreamfusion']
            elif Diff_kwargs.prompt_path.endswith('.parquet'):
                import pandas as pd
                prompts = pd.read_parquet(Diff_kwargs.prompt_path)['prompt'].tolist()
            dist.print0("Num of prompts: {}".format(len(prompts)))
            val_prompts = ["a photograph of an astronaut riding a horse", "a DSLR photo of a cat jumping over a fence, high-res", "a poodle wearing a baseball cap and holding a dictionary in hand", "a blue Porsche 356 parked in front of a brick wall"]
        else:
            prompts = ['a chimpanzee holding a peeled banana']
            val_prompts = prompts

        with torch.no_grad():
            text_embeds, negative_embeds = pipe.encode_prompt(val_prompts, num_images_per_prompt=1, device=device, do_classifier_free_guidance=True)
            text_embeddings_vis = torch.cat([negative_embeds, text_embeds])
            if gen_type == 'sin':
                text_embeddings = text_embeddings_vis
    elif arch == 'unet_zero123plus':
        from safetensors.torch import load_file
        # state_dict = load_file('/home/chenwang/workspace/zero123plusft/logs/zero123pluswhite9k.safetensors', device='cpu')
        # state_dict1 = {}
        # for keys in state_dict.keys():
        #     if keys.startswith('unet.unet.'):
        #         state_dict1[keys[10:]] = state_dict[keys]
        from huggingface_hub import hf_hub_download
        # unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        unet_ckpt_path = '/home/xuyimeng/Repo/InstantMesh/ckpts/diffusion_pytorch_model_v11.bin'
        state_dict1 = torch.load(unet_ckpt_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict1, strict=True)
        pipe.prepare()
        with torch.no_grad():
            cond = to_rgb_image(Image.open('./data/lysol.png'))
            text_embeddings, cross_attention_kwargs = pipe.prepare_conditions(cond, guidance_scale=Diff_kwargs.cfg_tchr)
            batch_val = max(2, batch_gpu) # at least two different latents
            text_embeddings = torch.cat([text_embeddings[:1].repeat(batch_val, 1, 1), text_embeddings[1:2].repeat(batch_val, 1, 1)])
            text_embeddings_vis = text_embeddings.clone()
            cond_lat = cross_attention_kwargs['cond_lat']
            # empty image and condition image
            cond_lat = torch.cat([cond_lat[:1].repeat(batch_val, 1, 1, 1), cond_lat[1:2].repeat(batch_val, 1, 1, 1)])
            cross_attention_kwargs_vis = {'cond_lat': cond_lat}
            cross_attention_kwargs_stu = cross_attention_kwargs_vis
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif arch == 'unet_zero123':
        pipe.prepare_image_embeddings()
        image_camera_embeddings = pipe.get_image_camera_embeddings(torch.tensor([30]).to(device), torch.tensor([-50]).to(device), torch.tensor([1.2]).to(device)).clone()
        image_latents = pipe.image_latents.clone()
        
    Diff = pipe.unet.train().requires_grad_(True).to(device)
    Diff_pre = copy.deepcopy(Diff).eval().requires_grad_(False).to(device)
    Diff_params = Diff.parameters()
    vae = pipe.vae.requires_grad_(False).eval().to(device) if Diff_kwargs.latent else None
    Diff_params = Diff.parameters()
    if not Diff_kwargs.ft_all:
        cross_attention_kwargs_stu = {'scale': 1.0}
        Diff, Diff_lora = extract_lora_diffusers(Diff, device)
        Diff_params = Diff_lora.parameters()
    scheduler = pipe.scheduler
    torch.cuda.empty_cache()

    if gen_type == 'sin':
        all_images = torch.nn.Parameter(torch.rand(1, 4, int(ratio*resolution), resolution, device=device))
        G_params = [all_images]
    elif gen_type == 'LRM':
        lrm_inferer = LRMInferrer(device)
        image = Image.open('./data/lysol.png')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        # if RGBA, blend to RGB
        if image.shape[1] == 4:
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(lrm_inferer.infer_kwargs['source_size'], lrm_inferer.infer_kwargs['source_size']), mode='bicubic', align_corners=True).clamp(0, 1)

        G = lrm_inferer.get_single_image_generator(image, [0], [0], lrm_inferer.infer_kwargs['render_size']).requires_grad_(True)
        G_frozen = copy.deepcopy(G).requires_grad_(False)
        G_params = G.parameters()
        eval_cameras = lrm_inferer._default_render_cameras(1).to(device)
        zero123plus_cameras = lrm_inferer.pose_from_elevation_and_azimuth([30, 30, 30, -20, -20, -20], [30, 150, 270, 90, 210, 330])
        zero123plus_cameras = lrm_inferer.extrinsic_to_pose(zero123plus_cameras).to(device)
        frames = G.render_video(eval_cameras, render_size=256)
        save_video(frames, os.path.join(run_dir, 'init.mp4'))

        train_dataset = RandomCameraIterableDataset(parse_structured(RandomCameraDataModuleConfig, {'batch_size': batch_gpu}))
        train_loader = DataLoader(train_dataset, batch_size=None, collate_fn=train_dataset.collate)
        pose_iterator = iter(train_loader)
    else:
        G = copy.deepcopy(Diff).requires_grad_(True).train().to(device)
        G.is_generator = True
        G_params = G.parameters()
        grid_z = torch.randn([text_embeddings_vis.shape[0]//2, num_channels, int(ratio * resolution), resolution], device=device, dtype=dtype)
        with torch.no_grad():
            G.eval()
            out = predict_x0(G, grid_z, text_embeddings_vis, t=Diff_kwargs.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs_vis, scheduler=scheduler, model=arch)
        save_image_grid(out.clone().detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'init_latents.png'), drange=[-1, 1], grid_size=get_grid_size(out.shape[0], 8))
        if Diff_kwargs.latent:
            image = decode_latents(out, vae, arch == 'unet_zero123plus')
            save_image_grid(image.clone().detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'init_img.png'), drange=[-1, 1], grid_size=get_grid_size(out.shape[0], 8))

    dist.print0('Setting up optimizer...')
    G_opt = dnnlib.util.construct_class_by_name(params=G_params, **G_opt_kwargs) # subclass of torch.optim.Optimizer
    Diff_opt = dnnlib.util.construct_class_by_name(params=Diff_params, **Diff_opt_kwargs) # subclass of torch.optim.Optimizer

    # Resume from existing pickle.
    if resume_pkl is not None:
        print(f'Resuming from "{resume_pkl}"')
        resume_data = pickle.load(open(resume_pkl, 'rb'))
        for name, module in [('Diff', Diff), ('G', G)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        print(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        G_opt.load_state_dict(data['G_opt_state'])
        Diff_opt.load_state_dict(data['Diff_opt_state'])
        cur_tick = data['cur_tick']
        del data # conserve memory

    # Distribute across GPUs.
    dist.print0(f'Distributing across {num_gpus} GPUs...')
    modules = [Diff] if gen_type == 'sin' else [Diff, G]
    for module in modules:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    dist.print0('Setting up training phases...')

    # Initialize logs.
    dist.print0('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, f'stats-{curr_time}.jsonl'), 'wt')
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(run_dir)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...\n')
    cur_tick = cur_tick if resume_pkl is not None else 0
    cur_nimg = cur_tick * batch_size
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    
    def optimizer_step(params, opt):
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        opt.step()

    for i in tqdm(range(cur_tick, cur_tick + max_iters)):

        G_opt.zero_grad(set_to_none=True)
        Diff_opt.zero_grad(set_to_none=True)
        labels = torch.eye(Diff.label_dim, device=device)[torch.randint(Diff.label_dim, size=[batch_gpu], device=device)] if label_dim > 0 else None

        if gen_type == 'sin':
            if loss_type == 'sds':
                particles = all_images
            else:
                particles = all_images[torch.randint(0, n_images, [batch_size])]
        elif gen_type == 'LRM':
            G.train()
            if arch == 'unet_zero123' or arch == 'unet_sd':
                data = next(pose_iterator)
                c2w = data['c2w'][:, :3, :]
                render_cameras = lrm_inferer.extrinsic_to_pose(c2w).to(device)
                image_camera_embeddings = pipe.get_image_camera_embeddings(data['elevation'].to(device), data['azimuth'].to(device), torch.full_like(data['elevation'], 1.2).to(device))
                renderings = G(render_cameras)
                renderings = F.interpolate(renderings, (256, 256), mode="bilinear", align_corners=False) * 2.0 - 1.0
                particles = vae.encode(renderings).latent_dist.sample() * vae.config.scaling_factor
            elif arch == 'unet_zero123plus':
                render_cameras = zero123plus_cameras.clone()
                renderings = G(render_cameras)
                renderings = einops.rearrange(renderings, '(w2 h2) c h w -> c (h2 h) (w2 w)', h2=3, w2=2).unsqueeze(0)
                renderings = F.interpolate(renderings, (960, 640), mode="bilinear", align_corners=False) * 2.0 - 1.0
                particles = vae.encode(renderings).latent_dist.sample()
        else:
            z = torch.randn([batch_gpu, num_channels, int(ratio*resolution), resolution], device=device, dtype=dtype)
            G.train()
            data = next(dataset_iterator)
            if arch == 'unet_zero123plus':
                text_embeddings, cross_attention_kwargs = pipe.prepare_conditions(data['img'], guidance_scale=Diff_kwargs.cfg_tchr)
                # empty image and condition image
                cross_attention_kwargs_stu = cross_attention_kwargs
            else:
                prompts = data
                with torch.no_grad():
                    text_embeds, negative_embeds = pipe.encode_prompt(prompts, num_images_per_prompt=1, device=device, do_classifier_free_guidance=True)
                    text_embeddings = torch.cat([negative_embeds, text_embeds])
                indices = torch.randperm(text_embeds.shape[0])[:batch_gpu]
                text_embeddings = torch.cat([negative_embeds[indices], text_embeds[indices]])
            particles = predict_x0(G, z, text_embeddings, t=Diff_kwargs.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, model=arch)


        # Execute training loop.
        # (1) Train Generator
        with torch.no_grad():
            y, augment_labels = particles, None
            
            t = (((0.02 - 0.98) * torch.rand([batch_gpu], device=device) + 0.98) * 999).long()
            n = torch.randn_like(y) 
            y_noisy = scheduler.add_noise(y, n, t)
            if arch == 'unet_zero123':
                n_pred = pipe.forward_unet(Diff_pre, image_latents, image_camera_embeddings, Diff_kwargs.cfg_tchr, y_noisy, t, encoder_hidden_states=image_camera_embeddings)
            else:
                n_pred = predict_noise0_diffuser(Diff_pre, y_noisy, text_embeddings, t, guidance_scale=Diff_kwargs.cfg_tchr, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, model=arch)
            if loss_type == 'vsd':
                if arch == 'unet_zero123':
                    n_est = pipe.forward_unet(Diff, image_latents, image_camera_embeddings, Diff_kwargs.cfg_stu, y_noisy, t, encoder_hidden_states=image_camera_embeddings, cross_attention_kwargs=cross_attention_kwargs_stu)
                else:
                    n_est = predict_noise0_diffuser(Diff, y_noisy, text_embeddings, t, guidance_scale=Diff_kwargs.cfg_stu, cross_attention_kwargs=cross_attention_kwargs_stu, scheduler=scheduler, model=arch)
                grad = n_pred - n_est
            else:
                grad = n_pred - n

            grad = torch.nan_to_num(grad)
        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (particles - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(particles, target, reduction="sum") / particles.shape[0]

        loss_vsd.backward()
        G_params_post = [param for param in (G.parameters() if gen_type != 'sin' else [all_images]) if param.numel() > 0 and param.grad is not None]
        optimizer_step(G_params_post, G_opt)
        
        # # (2) Train Diffusion
        if loss_type == 'sds':
            loss_diff = 0.0
        else:
            particles = particles.clone().detach() # detach from generator graph
            n_phi = torch.randn_like(particles)
            t_phi = (((0.02 - 0.98) * torch.rand([batch_gpu], device=device) + 0.98) * 999)
            t_phi = t_phi if arch == 'unet_edm' else t_phi.long()
            x_noisy = scheduler.add_noise(particles, n_phi, t_phi).detach()
            if arch == 'unet_zero123':
                n_phi_pred = pipe.forward_unet(Diff, image_latents, image_camera_embeddings.clone().detach(), Diff_kwargs.cfg_stu, x_noisy, t_phi, cross_attention_kwargs=cross_attention_kwargs_stu)  
            else:
                n_phi_pred = predict_noise0_diffuser(Diff, x_noisy, text_embeddings, t_phi, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs_stu, scheduler=scheduler, model=arch)
            loss_diff = torch.nn.functional.mse_loss(n_phi_pred, n_phi)
            loss_diff.sum().mul(1/batch_gpu).backward()

        Diff_params_post = [param for param in (Diff_lora.parameters() if not Diff_kwargs.ft_all else Diff.parameters()) if param.numel() > 0 and param.grad is not None]
        optimizer_step(Diff_params_post, Diff_opt)
        # for (params, opt) in [(G_params_post, G_opt), (Diff_params_post, Diff_opt)]:
        #     if len(params) > 0:
        #         flat = torch.cat([param.grad.flatten() for param in params])
        #         if num_gpus > 1:
        #             torch.distributed.all_reduce(flat)
        #             flat /= num_gpus
        #         misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
        #         grads = flat.split([param.numel() for param in params])
        #         for param, grad in zip(params, grads):
        #             param.grad = grad.reshape(param.shape)
        #     opt.step()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1
        cur_tick += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000) or (cur_tick == max_iters)
        # if (not done) and (cur_tick != 0) and (cur_tick % image_snapshot_ticks != 0):
        #     continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        training_stats.report0('Progress/tick', cur_tick)
        training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time)
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        training_stats.report0('Loss/vsd', loss_vsd)
        training_stats.report0('Loss/diff', loss_diff)
        training_stats.report0('lr/Diff', Diff_opt.param_groups[0]['lr'])
        training_stats.report0('lr/G', G_opt.param_groups[0]['lr'])
        # if rank == 0:
        #     print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            dist.print0('\nAborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                if gen_type == 'sin':
                    images = all_images.clone().detach()
                    save_image_grid(images.clamp(-1, 1).cpu(), os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1, 1], grid_size=(n_images, 1))
                    if Diff_kwargs.latent:
                        img = unscale_latents(images)
                        img = vae.decode(images[:4]/vae.config.scaling_factor).sample
                        img = unscale_image(img)
                        save_image_grid(img.detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'{cur_tick:06d}-img.png'), drange=[-1, 1], grid_size=(n_images, 1))
                elif gen_type == 'LRM':
                    G.eval()
                    frames = G.render_video(render_cameras)
                    imgs = pipe.gen_from_cond(image_latents, image_camera_embeddings, Diff_pre)
                    frames_init = G_frozen.render_video(render_cameras)
                    frames = [torch.cat([frame_init, frame, img], axis=-1) for (i, (frame_init, frame, img)) in enumerate(zip(frames_init, frames, imgs))]

                    save_video(frames, os.path.join(run_dir, f'{cur_tick:06d}.mp4'))

                    save_image_grid(G.plane.clone().detach().cpu()[0][:, -3:, :, :], os.path.join(run_dir, f'{cur_tick:06d}-triplane.png'), drange=[-10, 10], grid_size=(3, 1))

                    if cur_tick % 500 == 0:
                        frames = G.render_video(eval_cameras)
                        save_video(frames, os.path.join(run_dir, f'{cur_tick:06d}_orbit.mp4'))
                elif arch.startswith('unet'):
                    G.eval()
                    out = predict_x0(G, grid_z, text_embeddings_vis, t=Diff_kwargs.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs_vis, scheduler=scheduler, model=arch) if Diff_kwargs.cond != None else predict_x0_ldm(G, grid_z, t=Diff_kwargs.init_t, scheduler=scheduler, labels=grid_labels)
                    if arch.startswith('unet') and Diff_kwargs.latent:
                        img = decode_latents(out, vae, arch=='unet_zero123plus')
                        save_image_grid(img.clone().detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'{cur_tick:06d}-img.png'), drange=[-1, 1], grid_size=get_grid_size(out.shape[0], 8))
                        out /= vae.config.scaling_factor
                    save_image_grid(out.clone().detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1, 1], grid_size=get_grid_size(out.shape[0], 8))

                    out = predict_x0(G, z, text_embeddings, t=Diff_kwargs.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, model=arch) if Diff_kwargs.cond != None else predict_x0_ldm(G, z, t=Diff_kwargs.init_t, scheduler=scheduler, labels=labels)
                    if Diff_kwargs.latent:
                        img = decode_latents(out, vae, arch=='unet_zero123plus')
                    save_image_grid(img.clone().detach().clamp(-1, 1).cpu(), os.path.join(run_dir, f'{cur_tick:06d}-rand.png'), drange=[-1, 1], grid_size=get_grid_size(out.shape[0], 8))
                    save_image_grid(data['img'].clone().detach().permute(0,3,1,2).clamp(0, 255).cpu(), os.path.join(run_dir, f'{cur_tick:06d}-cond.png'), drange=[0, 255], grid_size=get_grid_size(out.shape[0], 8))
        torch.cuda.empty_cache()

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(dataset_kwargs=dict(dataset_kwargs))
            for name, module in [('Diff', Diff)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            if gen_type == 'sin':
                snapshot_data['all_img'] = all_images.clone().detach().cpu()
            else:
                snapshot_data['G'] = copy.deepcopy(G).eval().requires_grad_(False).cpu()
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_tick:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
            
            torch.save(dict(G_opt_state=G_opt.state_dict(), Diff_opt_state=Diff_opt.state_dict(), cur_tick=cur_tick), os.path.join(run_dir, f'training-state-{cur_tick:06d}.pt'))

        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_tfevents is not None:
            global_step = cur_tick
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0('\nExiting...')

#----------------------------------------------------------------------------
