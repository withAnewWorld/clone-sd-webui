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
import math

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

opt_C = 4
opt_f = 8


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
        default="latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latent-diffusion/models/ldm/txt2img-f8-large/model.ckpt",
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


def image_grid(imgs, batch_size):
    if opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = round(math.sqrt(len(imgs)))

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color="black")

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


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
):
    torch.cuda.empty_cache()

    outpath = opt.outdir or "outputs/img2img-samples"

    if seed == -1:
        seed = random.randrange(4294967294)

    sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

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
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            init_image = 2.0 * image - 1.0
            init_image = init_image.to(opt.device)
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

                        if not opt.skip_save:
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

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
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


def dream(
    prompt: str,
    ddim_steps: int,
    sampler_name: str,
    use_GFPGAN: bool,
    ddim_eta: float,
    n_iter: int,
    n_samples: int,
    cfg_scale: float,
    seed: int,
    height: int,
    width: int,
):
    torch.cuda.empty_cache()

    outpath = opt.outdir or "outputs/txt2img-samples"

    if seed == -1:
        seed = random.randrange(4294967294)

    seed = int(seed)

    is_PLMS = sampler_name == "PLMS"
    is_DDIM = sampler_name == "DDIM"
    is_Kdif = sampler_name == "k-diffusion"

    sampler = None
    if is_PLMS:
        sampler = PLMSSampler(model)
    elif is_DDIM:
        sampler = DDIMSampler(model)
    elif is_Kdif:
        pass
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    model_wrap = K.external.CompVisDenoiser(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        for n in range(n_iter):
            for batch_index, prompts in enumerate(data):
                uc = None
                if cfg_scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [opt_C, height // opt_f, width // opt_f]

                current_seed = seed + n * len(data) + batch_index
                torch.manual_seed(current_seed)

                if is_Kdif:
                    sigmas = model_wrap.get_sigmas(ddim_steps)
                    x = (
                        torch.randn([n_samples, *shape], device=opt.device) * sigmas[0]
                    )  # for GPU draw
                    model_wrap_cfg = CFGDenoiser(model_wrap)
                    samples_ddim = K.sampling.sample_lms(
                        model_wrap_cfg,
                        x,
                        sigmas,
                        extra_args={"cond": c, "uncond": uc, "cond_scale": cfg_scale},
                        disable=False,
                    )

                elif sampler is not None:
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=None,
                    )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                if not opt.skip_save or not opt.skip_grid:
                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        x_sample = x_sample.astype(np.uint8)

                        # if use_GFPGAN and GFPGAN is not None:
                        #     cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                        #     x_sample = restored_img

                        image = Image.fromarray(x_sample)

                        image.save(
                            os.path.join(
                                sample_path,
                                f"{base_count:05}-{current_seed}_{prompt.replace(' ', '_')[:128]}.png",
                            )
                        )
                        output_images.append(image)
                        base_count += 1

        if not opt.skip_grid:
            # additionally, save as grid
            grid = image_grid(output_images, batch_size)
            grid.save(os.path.join(outpath, f"grid-{grid_count:04}.png"))
            grid_count += 1

    if sampler is not None:
        del sampler

    info = f"""
{prompt}
Steps: {ddim_steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN else ''}
    """.strip()

    return output_images, seed, info


if __name__ == "__main__":
    opt = parse_args()

    img = Image.open(opt.image)
    model = get_model(opt.config, opt.ckpt)
    model = model.half().to(opt.device)

    dream_interface = gr.Interface(
        dream,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="a painting of a virus monster playing guitar",
                lines=1,
            ),
            gr.Slider(
                minimum=1, maximum=150, step=1, label="Sampling Steps", value=150
            ),
            gr.Radio(
                label="Sampling method",
                choices=["DDIM", "PLMS", "k-diffusion"],
                value="PLMS",
            ),
            # gr.Checkbox(label='Enable Fixed Code sampling', value=False),
            gr.Checkbox(label="Fix faces using GFPGAN", value=False, visible=False),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="DDIM ETA",
                value=0.0,
                visible=False,
            ),
            gr.Slider(
                minimum=1, maximum=16, step=1, label="Sampling iterations", value=1
            ),
            gr.Slider(
                minimum=1, maximum=4, step=1, label="Samples per iteration", value=1
            ),
            gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.5,
                label="Classifier Free Guidance Scale",
                value=7.0,
            ),
            gr.Number(label="Seed", value=-1),
            gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
            gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
        ],
        outputs=[
            gr.Gallery(label="Images"),
            gr.Number(label="Seed"),
            gr.Textbox(label="Copy-paste generation parameters"),
        ],
        title="Stable Diffusion Text-to-Image K",
        description="Generate images from text with Stable Diffusion (using K-LMS)",
        allow_flagging="never",
    )

    img2img_interface = gr.Interface(
        translation,
        inputs=[
            gr.Textbox(
                placeholder="A fantasy landscape, trending on artstation.", lines=1
            ),
            gr.Image(
                value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                source="upload",
                interactive=True,
                type="pil",
            ),
            gr.Slider(
                minimum=1, maximum=150, step=1, label="Sampling Steps", value=150
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="DDIM ETA",
                value=0.0,
                visible=False,
            ),
            gr.Slider(
                minimum=1, maximum=50, step=1, label="Sampling iterations", value=1
            ),
            gr.Slider(
                minimum=1, maximum=8, step=1, label="Samples per iteration", value=1
            ),
            gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.5,
                label="Classifier Free Guidance Scale",
                value=7.0,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                label="Denoising Strength",
                value=0.75,
            ),
            gr.Number(label="Seed", value=-1),
            gr.Slider(
                minimum=64, maximum=2048, step=64, label="Resize Height", value=512
            ),
            gr.Slider(
                minimum=64, maximum=2048, step=64, label="Resize Width", value=512
            ),
        ],
        outputs=[gr.Gallery(), gr.Number(label="Seed")],
        title="Stable Diffusion Image-to-Image",
        description="Generate images from images with Stable Diffusion",
    )

    demo = gr.TabbedInterface(
        interface_list=[dream_interface, img2img_interface],
        tab_names=["Dream", "Image Translation"],
    )

    demo.launch()
