import gradio as gr
import os
import csv
from PIL import Image


class Flagging(gr.FlaggingCallback):
    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None) -> int:
        os.makedirs("log/images", exist_ok=True)

        # Those must match the "dream" function
        (
            prompt,
            ddim_steps,
            sampler_name,
            use_GFPGAN,
            prompt_matrix,
            ddim_eta,
            n_iter,
            n_samples,
            cfg_scale,
            requested_seed,
            height,
            width,
            images,
            seed,
            comment,
        ) = flag_data

        filenames = []

        with open("log/log.csv", "a", encoding="utf8", newline="") as file:
            import time

            at_start = file.tell() == 0
            writer = csv.writer(file)
            if at_start:
                writer.writerow(
                    ["prompt", "seed", "width", "height", "cfgs", "steps", "filename"]
                )

            filename_base = str(int(time.time() * 1000))
            for i, filedata in enumerate(images):
                filename = (
                    f"log/images/{filename_base}"
                    + ("" if len(images) == 1 else "-" + str(i + 1))
                    + ".png"
                )
                filenames.append(filename)

                image = Image.open(filedata["image"]["path"])
                new_path = os.path.join(os.getcwd(), filename)
                image.save(new_path)

            writer.writerow(
                [prompt, seed, width, height, cfg_scale, ddim_steps, filenames[0]]
            )
        print("Logged: ", filenames[0])
