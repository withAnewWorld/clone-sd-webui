import os
from PIL import Image
import argparse
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
import mimetypes
import random

import k_diffusion as K
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torch

from utils import create_random_tensors, get_model, draw_prompt_matrix
import math

from gr_utils import Flagging

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

opt_C = 4
opt_f = 8

invalid_filename_chars = '<>:"/\|?*\n'


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
        default=-1,
        help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="v1-5-pruned-emaonly.ckpt.1",
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


def image_grid(imgs, batch_size, round_down=False):
    if opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = int(rows) if round_down else round(rows)

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


class KDiffusionSampler:
    def __init__(self, m):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)

    def sample(
        self,
        S,
        conditioning,
        batch_size,
        shape,
        verbose,
        unconditional_guidance_scale,
        unconditional_conditioning,
        eta,
        x_T,
    ):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.sample_lms(
            model_wrap_cfg,
            x,
            sigmas,
            extra_args={
                "cond": conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
            },
            disable=False,
        )

        return samples_ddim, None


def img2img(
    prompt: str,
    init_img,
    ddim_steps: int,
    prompt_matrix: bool,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    denoising_strength: float,
    seed: int,
    height: int,
    width: int,
):
    outpath = opt.outdir or "outputs/img2img-samples"

    sampler = KDiffusionSampler(model)

    assert 0.0 <= denoising_strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(denoising_strength * ddim_steps)

    def init():
        image = init_img.convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        init_image = 2.0 * image - 1.0
        init_image = init_image.to(opt.device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space
        return (init_latent,)

    def sample(init_data, x, conditioning, unconditional_conditioning):
        (x0,) = init_data
        sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
        noise = x * sigmas[ddim_steps - t_enc - 1]

        xi = x0 + noise
        sigma_sched = sigmas[ddim_steps - t_enc - 1 :]
        model_wrap_cfg = CFGDenoiser(sampler.model_wrap)

        sampler_ddim = K.sampling.sample_lms(
            model_wrap_cfg,
            xi,
            sigma_sched,
            extra_args={
                "cond": conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": cfg_scale,
            },
            disable=False,
        )
        return sampler_ddim

    output_images, seed, info = process_images(
        outpath=outpath,
        func_init=init,
        func_sample=sample,
        prompt=prompt,
        seed=seed,
        sampler_name="k-diffusion",
        batch_size=batch_size,
        n_iter=n_iter,
        steps=ddim_steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=False,
    )

    del sampler

    return output_images, seed, info


def process_images(
    outpath,
    func_init,
    func_sample,
    prompt,
    seed,
    sampler_name,
    batch_size,
    n_iter,
    steps,
    cfg_scale,
    width,
    height,
    prompt_matrix,
    use_GFPGAN,
):
    """this is the main loop that both txt2img and img2img use
    it calls func_init once inside all the scopes and func_sample once per batch"""
    assert prompt is not None
    torch.cuda.empty_cache()

    if seed == -1:
        seed = random.randrange(4294967294)

    seed = int(seed)
    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "sample")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    prompt_matrix_parts = []
    if prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            current = prompt_matrix_parts[0]

            for n, text in enumerate(prompt_matrix_parts[1:]):
                if combination_num & (2**n) > 0:
                    current += ("" if text.strip().startswith(",") else ", ") + text
            all_prompts.append(current)

        n_iter = math.ceil(len(all_prompts) / batch_size)
        all_seeds = len(all_prompts) * [seed]
        print(
            f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches."
        )
    else:
        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        init_data = func_init()

        for n in range(n_iter):
            batch_slice = slice(n * batch_size, (n + 1) * batch_size)
            prompts = all_prompts[batch_slice]
            seeds = all_seeds[batch_slice]

            uc = n
            if cfg_scale != 1.0:
                uc = model.get_learned_conditioning(len(prompts) * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)

            x = create_random_tensors(
                [opt_C, height // opt_f, width // opt_f], seeds=seeds, device=opt.device
            )

            samples_ddim = func_sample(
                init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if prompt_matrix or not opt.skip_save or not opt.skip_grid:
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    x_sample = x_sample.astype(np.uint8)

                image = Image.fromarray(x_sample)
                filename = f"{base_count:05}-{seeds[i]}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"

                image.save(os.path.join(sample_path, filename))

                output_images.append(image)
                base_count += 1

        if prompt_matrix or not opt.skip_grid:
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)

            if prompt_matrix:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    import sys

                    print("Error creating prompt_matrix text: ", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)
            grid.save(os.path.join(outpath, f"grid-{grid_count:04}.png"))
            grid_count += 1
    info = f"""
{prompt}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}
        """.strip()

    return output_images, seed, info


def txt2img(
    prompt: str,
    ddim_steps: int,
    sampler_name: str,
    use_GFPGAN: bool,
    prompt_matrix: bool,
    ddim_eta: float,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    seed: int,
    height: int,
    width: int,
):
    outpath = opt.outdir or "outputs/txt2img-samples"

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

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning):
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=conditioning,
            batch_size=int(x.shape[0]),
            shape=x[0].shape,
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=unconditional_conditioning,
            eta=ddim_eta,
            x_T=x,
        )
        return samples_ddim

    output_images, seed, info = process_images(
        outpath=outpath,
        func_init=init,
        func_sample=sample,
        prompt=prompt,
        seed=seed,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=ddim_steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=use_GFPGAN,
    )
    del sampler

    return output_images, seed, info


if __name__ == "__main__":
    opt = parse_args()

    img = Image.open(opt.image)
    model = get_model(opt.config, opt.ckpt)
    model = model.half().to(opt.device)

    txt2img_interface = gr.Interface(
        txt2img,
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
            gr.Checkbox(
                label="Create prompt matrix (separate multiple prompts using |, and get all combinations of them)",
                value=False,
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
                minimum=1,
                maximum=16,
                step=1,
                label="Batch count (how many batches of images to generate)",
                value=1,
            ),
            gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                label="Batch size (how many images are in a batch; memory-hungry)",
                value=1,
            ),
            gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.5,
                label="Classifier Free Guidance Scale(how strongly should the image follow the prompt)",
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
        flagging_callback=Flagging(),
    )

    img2img_interface = gr.Interface(
        img2img,
        inputs=[
            gr.Textbox(
                placeholder="A fantasy landscape, trending on artstation.", lines=1
            ),
            gr.Image(
                value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                # source="upload",
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
        outputs=[
            gr.Gallery(),
            gr.Number(label="Seed"),
            gr.Textbox(label="Copy-Paste generation parameters"),
        ],
        title="Stable Diffusion Image-to-Image",
        description="Generate images from images with Stable Diffusion",
        allow_flagging="never",
    )

    demo = gr.TabbedInterface(
        interface_list=[txt2img_interface, img2img_interface],
        tab_names=["txt2img", "img2img"],
    )

    demo.launch()
