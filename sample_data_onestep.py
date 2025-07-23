import sys
sys.path.append('../')
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

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
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

class ObjaverseLVISSampleData(Dataset):
    def __init__(self, path, source_size=256, low=0, high=10000):
        self.root_dir = path
        self.scenes = sorted(os.listdir(path))[low:high]
        self.source_size = source_size

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # return img and pose
        # filename = self.paths[index]
        filename = os.path.join(self.root_dir, self.scenes[idx])
        z = torch.from_numpy(np.load(os.path.join(self.root_dir, self.scenes[idx], 'z.npy')))
        cond = np.array(Image.open( os.path.join(self.root_dir, self.scenes[idx], 'cond.png') ))
        return {
            'z': z,
            'cond': cond,
            'path': filename.split('/')[-1]
        }

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def generate(opt):
    bs = opt.sample_bs
    nv = opt.sample_nv

    device = torch.device("cuda")

    # torch.cudnn.benchmark = True
    dataset = ObjaverseLVISSampleData(opt.sample_data_path, low=opt.sample_start, high=opt.sample_end)
    output_path = opt.sample_output_path

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
        pipeline.prepare()
        with tqdm.tqdm(loader) as pbar:
            for data in pbar:
                if 'z' in data:
                    images = inferrer.onestep_gen(data['z'].to(device), data['cond'].to(device)).clip(0, 255)
                else:
                    z = torch.randn([bs, 4, 120, 80], device=device, dtype=torch.float16)
                    images = inferrer.onestep_gen(z, data['cond'].to(device)).clip(0, 255)
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                for i, img in enumerate(images):
                    name = data["path"][i]
                    os.makedirs(os.path.join(output_path, name), exist_ok=True)
                    Image.fromarray(data['cond'][i].cpu().numpy().astype(np.uint8)).save(os.path.join(output_path, f'{name}/cond.png'))
                    np.save(os.path.join(output_path, f'{data["path"][i]}/z.npy'), z[i].cpu().numpy())
                    img = Image.fromarray(img).save(os.path.join(output_path, f'{data["path"][i]}/onestep-20k.png'))
                torch.cuda.empty_cache()


if __name__ == "__main__":

    opt = tyro.cli(AllConfigs)
    generate(opt)