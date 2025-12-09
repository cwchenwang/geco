import tyro
import time
import random

import torch
from src.options import AllConfigs
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from safetensors.torch import load_file
from src.data.dataset_noise import NoiseImageDataset
from src.model_mesh_geco import InstantMeshFT
import kiui
from datetime import datetime
import torch.utils.tensorboard as tensorboard

def main():    
    set_seed(42)
    opt = tyro.cli(AllConfigs)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # model
    model = InstantMeshFT(opt)
    opt.workspace += opt.model_type + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
    if accelerator.is_main_process:
        writer = tensorboard.SummaryWriter(opt.workspace)

    import shutil, os
    src_snapshot_folder = os.path.join(opt.workspace, 'src')
    ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
    for folder in ['src']:
        dst_dir = os.path.join(src_snapshot_folder, folder)
        shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
    import yaml
    with open(os.path.join(opt.workspace, 'config.yml'), 'w') as f:
        yaml.dump(tyro.extras.to_yaml(opt), f)

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    train_dataset = NoiseImageDataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = NoiseImageDataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                # scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                for k, v in out.items():
                    if k.startswith('loss'):
                        writer.add_scalar(f'train/{k}', v.item(), i + len(train_dataloader) * epoch)
                # writer.add_scalar('train/loss_lpips', out['loss_lpips'].item(), i + len(train_dataloader) * epoch)
                writer.add_scalar('train/psnr', out['psnr'].item(), i + len(train_dataloader) * epoch)
                # writer.add_scalar('train/loss_splatter', out['loss_sp'].item(), i + len(train_dataloader) * epoch)
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                # if i % 500 == 0:
                #     model.save_training_output(out, opt.workspace, epoch, i)
                
                # save log images
                if i % 500 == 0:
                    gt_images = out['images_gt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    student_images = out['images_student'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    student_images = student_images.transpose(0, 3, 1, 4, 2).reshape(-1, student_images.shape[1] * student_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_student_images_{epoch}_{i}.jpg', student_images)

                    teacher_images = out['images_teacher'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    teacher_images = teacher_images.transpose(0, 3, 1, 4, 2).reshape(-1, teacher_images.shape[1] * teacher_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_teacher_images_{epoch}_{i}.jpg', teacher_images)

                    if 'zero123plus' in out:
                        zero123plus_out = (out['zero123plus'].detach().cpu().numpy().clip(-1, 1) + 1) / 2 # (B, 3, 960, 640)
                        zero123plus_out = zero123plus_out.transpose(2, 0, 3, 1).reshape(zero123plus_out.shape[2], -1, 3)
                        kiui.write_image(f'{opt.workspace}/train_pred_zero123_images_{epoch}_{i}.jpg', zero123plus_out)
                    torch.cuda.empty_cache()
                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            writer.add_scalar('train/loss', total_loss.item(), epoch)
            writer.add_scalar('train/psnr', total_psnr.item(), epoch)
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        if epoch % opt.save_freq == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, os.path.join(opt.workspace, str(epoch)))

        if epoch % opt.eval_freq == 0:
            # eval
            with torch.no_grad():
                model.eval()
                total_psnr = 0
                for i, data in enumerate(test_dataloader):

                    out = model(data)
        
                    psnr = out['psnr']
                    total_psnr += psnr.detach()
                    
                    # save some images
                    if accelerator.is_main_process:
                        gt_images = out['images_gt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                        student_images = out['images_student'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        student_images = student_images.transpose(0, 3, 1, 4, 2).reshape(-1, student_images.shape[1] * student_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_student_images_{epoch}_{i}.jpg', student_images)

                        # kiui.write_image(f'{opt.workspace}/eval_pred_splatter_{epoch}_{i}.png', out['sp_pred'])

                        # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

                torch.cuda.empty_cache()

                total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
                if accelerator.is_main_process:
                    total_psnr /= len(test_dataloader)
                    accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
                    writer.add_scalar('eval/psnr', total_psnr.item(), epoch)


if __name__ == "__main__":
    main()
