import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers
import math

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def extract_lora_diffusers(unet, device):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        ).to(device)
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # self.unet.requires_grad_(True)
    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)
    # self.params_to_optimize = unet_lora_layers.parameters()
    ### end lora
    return unet, unet_lora_layers

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=1.0, cross_attention_kwargs={}, scheduler=None, lora_v=False, model='sd'):
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
        cak = {'cond_lat': cross_attention_kwargs['cond_lat'][batch_size:]}
        # if cross_attention_kwargs is not None:
        #     for key in cross_attention_kwargs:
        #         if isinstance(cross_attention_kwargs[key], torch.Tensor):
        #             cak[key] = cross_attention_kwargs[key][batch_size:]
        #         else:
        #             cak[key] = cross_attention_kwargs[key]
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
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=device
    )
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    return alpha_t, sigma_t

def predict_x0(unet, noisy_latents, text_embeddings, t, guidance_scale=1.0, cross_attention_kwargs={}, scheduler=None, lora_v=False, model='sd'):
    alpha_t, sigma_t = get_alpha(scheduler, t, noisy_latents.device)
    noise_pred = predict_noise0_diffuser(unet, noisy_latents * sigma_t, text_embeddings, t=t, guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, model=model)
    return (noisy_latents - noise_pred) * sigma_t/ alpha_t

def predict_x0_ldm(unet, noisy_latents, t, scheduler, labels=None, encoder_hidden_states=None):
    if unet.__class__.__name__ in ['SongUNet', 'DhariwalUNet']:
        t = torch.as_tensor(t, device=noisy_latents.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)[:, None, None, None].repeat(noisy_latents.shape[0], 1, 1, 1).to(noisy_latents.device)
        if unet.__class__.__name__ == 'SongUNet':
            alpha_t, sigma_t = scheduler.alpha(t / 999), scheduler.sigma(t / 999)
            noise_pred = unet(noisy_latents * sigma_t, t.flatten(), labels)
        else:
            alpha_t, sigma_t = scheduler.alpha(t.long()), scheduler.sigma(t.long())
            noise_pred = unet(noisy_latents * sigma_t, t.to(torch.float32).flatten(), labels)[:, :3]
        return (noisy_latents - noise_pred) * sigma_t / alpha_t
    else:
        noise_pred = unet(noisy_latents, torch.tensor(t).to(torch.float32)).sample
        alpha_t, sigma_t = get_alpha(scheduler, torch.tensor(t).long(), noisy_latents.device)
    return (noisy_latents - noise_pred) * sigma_t / alpha_t

def obtain_mvdream_inputs(noise, t, actual_num_frames, camera, dtype, device, image=None, do_classifier_free_guidance=False, image_embeds_pos=None, image_embeds_neg=None, image_latents_pos=None, image_latents_neg=None, prompt_embeds_neg=None, prompt_embeds_pos=None):
    if not do_classifier_free_guidance:
        unet_inputs = {
            'x': torch.cat([noise]),
            'timesteps': torch.tensor([t] * actual_num_frames, dtype=dtype, device=device),
            'context': torch.cat([prompt_embeds_pos] * actual_num_frames),
            'num_frames': actual_num_frames,
            'camera': torch.cat([camera]),
        }
        if image is not None:
            unet_inputs['ip'] = torch.cat([image_embeds_pos] * actual_num_frames)
            unet_inputs['ip_img'] = torch.cat([image_latents_pos]) # no repeat
    else:
        unet_inputs = {
            'x': torch.cat([noise] * 2),
            'timesteps': torch.tensor([t] * actual_num_frames * 2, dtype=dtype, device=device),
            'context': torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames),
            'num_frames': actual_num_frames,
            'camera': torch.cat([camera] * 2),
        }
        if image is not None:
            unet_inputs['ip'] = torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
            unet_inputs['ip_img'] = torch.cat([image_latents_neg] + [image_latents_pos]) # no repeat
    return unet_inputs