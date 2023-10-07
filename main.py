import os
import PIL
from PIL import Image
import argparse, sys, glob
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import contextmanager, nullcontext
import mimetypes
import random

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torch

from utils import get_model

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default=None
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="latent-diffusion/configs/latent-diffusion/cin256-v2.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latent-diffusion/models/ldm/cin256-v2/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="accelerator")
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--gfpgan-dir", type=str, help="GFPGAN directory", default="./GFPGAN"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(os.getcwd(), "sketch-mountains-input.jpg"),
        help="image path to apply stable diffusion",
    )
    args = parser.parse_args()
    return args


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def translation(
    prompt: str,
    init_img,
    ddim_steps: int,
    ddim_eta: float,
    n_iter: int,
    n_samples: int,
    cfg_scale: float,
    denoising_strength: float,
    seed: int,
    height: int,
    width: int,
    outpath: str = "outputs/img2img-samples",
    device: str = "cuda",
):
    torch.cuda.empty_cache()

    if seed == -1:
        seed = random.randrange(4294967294)

    sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    seedit = 0

    image = init_img.convert("RGB")
    w, h = image.size
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    output_images = []
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            init_image = 2.0 * image - 1.0
            init_image = init_image.to(device)
            init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image)
            )  # move to latent space
            x0 = init_latent

            sampler.make_schedule(
                ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False
            )

            assert (
                0.0 <= denoising_strength <= 1.0
            ), "can only work with strength in [0.0, 1.0]"
            t_enc = int(denoising_strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")
            with model.ema_scope():
                all_samples = list()
                for n in range(n_iter):
                    for batch_index, prompts in enumerate(data):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        sigmas = model_wrap.get_sigmas(ddim_steps)

                        current_seed = seed + n * len(data) + batch_index
                        torch.manual_seed(current_seed)

                        noise = (
                            torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                        )  # for GPU draw
                        xi = x0 + noise
                        sigma_sched = sigmas[ddim_steps - t_enc - 1 :]
                        # x = torch.randn([n_samples, *shape]).to(device) * sigmas[0] # for CPU draw
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": cfg_scale}

                        samples_ddim = K.sampling.sample_lms(
                            model_wrap_cfg,
                            xi,
                            sigma_sched,
                            extra_args=extra_args,
                            disable=False,
                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        if not args.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                image.save(
                                    os.path.join(
                                        sample_path,
                                        f"{base_count:05}-{current_seed}_{prompt.replace(' ', '_')[:128]}.png",
                                    )
                                )
                                output_images.append(image)
                                base_count += 1
                                seedit += 1

                        if not args.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not args.skip_grid:
                    # additio
                    # nally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, f"grid-{grid_count:04}.png")
                    )
                    Image.fromarray(grid.astype(np.uint8))
                    grid_count += 1

    del sampler
    return output_images, seed


if __name__ == "__main__":
    args = parse_args()

    img = Image.open(args.image)
    model = get_model(args.config, args.ckpt)
    sampler = DDIMSampler(model)

    output, seed = translation(
        "A fantasy landscape, trending on artstation.",
        img,
        1,
        0.0,
        1,
        1,
        1.0,
        0.0,
        -1,
        64,
        64,
        device=args.device,
    )
