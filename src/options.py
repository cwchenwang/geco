import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### zero123plus
    model_type: str = 'LGM'
    model_path: str = 'sudo-ai/zero123plus-v1.1' 
    custom_pipeline: str ='./pipelines/zero123plus.py'
    bg: float = 0.5

    pretrained_as_init: bool = False

    ### dataset
    # data mode (only support s3 now)
    data_path: str = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 6
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### training
    stage: int = 1
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 2000
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 1e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    save_freq: int = 200
    eval_freq: int = 200
    resume_pkl: str = "/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs1/zero123plus-lvis/zero123plus-gpus1-batch1-same-vsd-20240301-192604-cond0_t950_lvis/network-snapshot-005000.pkl"
    resume_lgm: Optional[str] = None
    joint: bool = False

    ### testing
    test_path: Optional[str] = None
    pipeline: str = 'zero123plus1step'
    generate_mv: bool = False
    include_input: bool = False
    cond_t: int = 0
    init_t: int = 800
    config_path: str = 'configs/instant-mesh-base.yaml'
    zero123plus_v: str = 'v11'
    scale: float = 1.0
    export_texmap: bool = False
    save_video: bool = False
    distance: float = 4.5
    use_cond: bool = False
    output_dir: str = './results'
    test_path: str = './images'
    input_elevation: float = 0.0

    ## sampling
    sample_bs: int = 6
    sample_nv: int = 50
    sample_data_path: str = '/mnt/kostas-graid/sw/envs/chenwang/data/lvis_first'
    sample_output_path: str = '/mnt/kostas-graid/sw/envs/chenwang/data/lvis_sample1'
    sample_start: int = 0
    sample_end: int = 10000

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=8,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
