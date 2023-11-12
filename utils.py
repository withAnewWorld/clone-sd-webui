from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from PIL import Image, ImageFont, ImageDraw
import math
import os
import torch
import torch.nn as nn


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def get_model(cfg_path, ckpt):
    config = OmegaConf.load(cfg_path)
    model = load_model_from_config(config, ckpt)
    return model


def draw_prompt_matrix(im, width, height, all_prompts):
    def wrap(text, d, font, line_length):
        lines = [""]
        for word in text.split():
            line = f"{lines[-1]} {word}".strip()
            if d.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return "\n".join(lines)

    def draw_texts(pos, x, y, texts, sizes):
        for i, (text, size) in enumerate(zip(texts, sizes)):
            active = pos & (1 << i) != 0

            if not active:
                text = "\u0336".join(text) + "\u0336"

            d.multiline_text(
                (x, y + size[1] / 2),
                text,
                font=fnt,
                fill=color_active if active else color_inactive,
                anchor="mm",
                align="center",
            )

            y += size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype("arial.ttf", fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_top = height // 4
    pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

    cols = im.width // width
    rows = im.height // height

    prompts = all_prompts[1:]

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
    prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

    sizes_hor = [
        (x[2] - x[0], x[3] - x[1])
        for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]
    ]
    sizes_ver = [
        (x[2] - x[0], x[3] - x[1])
        for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]
    ]
    hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
    ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_height / 2

        draw_texts(col, x, y, prompts_horiz, sizes_hor)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_height / 2

        draw_texts(row, x, y, prompts_vert, sizes_ver)

    return result


def create_random_tensors(shape, seeds, device):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x


class TextualInversionLoader:
    def load_textual_inversion(self, tokenizer, token_embedding, path, filename):
        token, embedding_vector = self.process_file(path, filename)

        new_token_embedding = self.get_resized_token_embeddings(token_embedding, 1)

        tokenizer.add_tokens(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_token_embedding.weight.data[token_id] = embedding_vector
        print(f"Loaded textual inversion embedding for {token}")

        return tokenizer, new_token_embedding

    def process_file(self, path, filename):
        name = os.path.splitext(filename)[0]

        data = torch.load(path)
        param_dict = data["string_to_param"]
        if hasattr(param_dict, "_parameters"):
            param_dict = getattr(param_dict, "_parameters")
        assert len(param_dict) == 1, "embedding file has multiple terms in it"
        embedding_vector = next(iter(param_dict.items()))[1].reshape(768)
        return name, embedding_vector

    def get_resized_token_embeddings(self, embedding, num_new_tokens):
        old_num_tokens, old_embedding_dim = embedding.weight.shape

        # Creating new embedding layer with more entries
        new_embedding = nn.Embedding(old_num_tokens + num_new_tokens, old_embedding_dim)

        # Setting device and type accordingly
        new_embedding.to(
            embedding.weight.device,
            dtype=embedding.weight.dtype,
        )

        # Copying the old entries
        new_embedding.weight.data[:old_num_tokens, :] = embedding.weight.data[
            :old_num_tokens, :
        ]

        self._test_resized_embeddings(embedding, new_embedding)
        return new_embedding

    def _test_resized_embeddings(self, old_embedding, new_embedding):
        old_num_tokens, old_embedding_dim = old_embedding.weight.shape
        for idx in range(old_num_tokens):
            assert (
                torch.all(old_embedding.weight[idx] == new_embedding.weight[idx]).item()
                == True
            ), f"different embedding weight in index: {idx}"
