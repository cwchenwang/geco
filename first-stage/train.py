import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from torch_utils import training_stats
from torch_utils import custom_ops
from datetime import datetime
import json
from training import training_loop_3d

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, f'log-{c.curr_time}.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop_3d.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    c.run_dir = os.path.join(outdir, f'{desc}')

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=True)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges
#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--max_iters',    help='Number of Iterations', metavar='INT',                      type=click.IntRange(min=1), default=10000)
@click.option('--suffix',       help='suffix dir', default='', type=str)
@click.option('--timestamp',    help='Whether to use timestamp for log dir', type=bool, default=True)
@click.option('--triplane_bsz', help='Number of Triplanes', type=int, default=2)
@click.option('--root_dir',     help='Root dir of dataset', type=str, default=None, required=False)
@click.option('--batch_pose',   help='Num of poses for each triplane', type=int, default=6)

@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=10000, show_default=True)
@click.option('--img-snap',     help='How often to save image snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=False)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--resume_state_dump', help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--model_path', metavar='str', type=str, required=False)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--lr',           help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--arch',         help='Network architecture', metavar='Zero123',          type=str, default='unet_zero123', show_default=True)
@click.option('--gen',          help='Generator Type', metavar='2d|LRM',   type=str, default='LRM', show_default=True)
@click.option('--loss_type', metavar='sds|vsd', type=click.Choice(['vsd', 'sds']), default='sds', show_default=True)
@click.option('--cfg_stu', metavar='FLOAT', type=click.FloatRange(min=1.0), required=True, default=1.0)
@click.option('--cfg_tchr', metavar='FLOAT', type=click.FloatRange(min=1.0), required=True, default=7.5)
@click.option('--ft_all', metavar='BOOL', type=bool, required=True, default=False)
@click.option('--custom_pipeline', metavar='str', type=str, required=False, default=None)
@click.option('--prompt_path', metavar='PATH', type=str, required=False, default=None)
@click.option('--init_t', metavar='INT', type=click.IntRange(min=1.0), required=True, default=999)
@click.option('--dtype', metavar='str', type=str, required=False, default='fp32')

def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8, lr=opts.glr)
    c.Diff_kwargs = dnnlib.EasyDict()
    c.Diff_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    # Training set.
    # c.dataset_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    data = opts.data.split(':')
    dataset_name = data[0]
    c.dataset_kwargs = dnnlib.EasyDict(resolution=int(data[1]))
    c.resolution = c.dataset_kwargs.resolution
    c.dataset_kwargs.triplane_bsz = opts.triplane_bsz
    c.dataset_kwargs.root_dir = opts.root_dir
    c.dataset_kwargs.batch_pose = opts.batch_pose

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus
  
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.network_snapshot_ticks = opts.snap
    c.image_snapshot_ticks = opts.img_snap
    c.state_dump_ticks = opts.dump
    c.random_seed = c.dataset_kwargs.random_seed = opts.seed

    # Diffusion Network architecture.
    if opts.arch.startswith('unet'):
        # prompts = json.load(open('prompt_lib.json'))['dreamfusion'][:opts.num_prompt]
        c.Diff_kwargs.update(model_path = opts.model_path, cfg_stu=opts.cfg_stu, cfg_tchr=opts.cfg_tchr, ft_all=opts.ft_all, latent=True, cond='t2i', custom_pipeline=opts.custom_pipeline)
        if opts.arch == 'unet_sd':
            c.update(num_channels=4, resolution=64)
            c.Diff_kwargs.update(prompt_path=opts.prompt_path)
        elif opts.arch == 'unet_if':
            c.update(num_channels=3, resolution=64)
            c.Diff_kwargs.update(latent=False)
        elif opts.arch == 'unet_zero123plus' or opts.arch == 'unet_zero123':
            c.update(num_channels=4, ratio=1.5)
            c.Diff_kwargs.update(cond='image')
            c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ObjaverseLVISData', path=opts.root_dir)
            
        c.Diff_kwargs.init_t = opts.init_t
        c.Diff_kwargs.dtype = opts.dtype
 
    c.arch = opts.arch

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.gen_type = opts.gen

    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
    
    if opts.resume_state_dump is not None:
        c.resume_state_dump = opts.resume_state_dump
    c.max_iters = opts.max_iters

    # Description string.
    desc = f'{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-{c.gen_type}-{opts.loss_type}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    if opts.timestamp:
        desc += datetime.now().strftime("-%Y%m%d-%H%M%S")
    if opts.suffix:
        desc += '-' + opts.suffix
    
    # Resuming from existing log directory
    if os.path.exists(os.path.join(opts.outdir, desc)):
        import glob
        pkls = sorted(glob.glob(os.path.join(opts.outdir, desc, '*.pt')))
        if len(pkls) > 0:
            match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(pkls[-1]))
            c.resume_pkl = os.path.join(opts.outdir, desc, f'network-snapshot-{match.group(1)}.pkl')
            c.resume_state_dump = os.path.join(opts.outdir, desc, f'training-state-{match.group(1)}.pt')
            c.resume_kimg = int(match.group(1))
            c.ada_kimg = 100 # Make ADA react faster at the beginning.
            c.ema_rampup = None # Disable EMA rampup.
    
    c.network_pkl = opts.network_pkl
    c.curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    c.loss_type = opts.loss_type

    ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
    src_snapshot_folder = os.path.join(opts.outdir, desc, 'src')
    import shutil
    for folder in ['training', 'torch_utils']:
        dst_dir = os.path.join(src_snapshot_folder, folder)
        shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
