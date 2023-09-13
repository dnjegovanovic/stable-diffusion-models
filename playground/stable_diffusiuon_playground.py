import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import itertools
import math
import mediapy as media
from pathlib import Path
import json
import numpy as np
import torchvision

with open("./token.json", "r") as f:
    data_token = json.load(f)

from huggingface_hub import login

login(data_token["token"])


def plt_show_image(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def create_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Disable the safety checkers
    def dummy_checker(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy_checker

    return pipe


def create_simple_img(pipe, prompt, img_name, generator=None, num_inference_steps=25):
    if num_inference_steps is not None:
        data = pipe(
            prompt, generator=generator, num_inference_steps=num_inference_steps
        )
    else:
        data = pipe(prompt, generator=generator)
    image = data.images[0]
    image.save(f"./playground_imgs/{img_name}.png")
    plt_show_image(image)


def create_video(images, video_name):
    shape = (480, 640)
    output_path = f'./playground_imgs/{video_name}.mp4'
    with media.VideoWriter(output_path, shape=shape, fps=10) as w:
        for image in images:
            w.add_image(image)

def simple_generation_diffuser_step_vis():
    generator = torch.Generator("cuda").manual_seed(1024)

    prompt1 = "a lovely cat running in the desert in Van Gogh style, trending art."
    prompt2 = "a sleeping cat enjoying the sunshine."
    prompt1 = "a lovely cat running in the desert in Van Gogh style, trending art."
    pipe = create_pipe()

    # create_simple_img(pipe, prompt2, "nfprompt_sunshine_interference_cat", generator)

    image_reservoir = []
    latents_reservoir = []

    # visualize diffuser steps
    @torch.no_grad()
    def plot_show_callback(i, t, latents):
        latents_reservoir.append(latents.detach().cpu())
        image = pipe.vae.decode(1 / 0.18215 * latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        # plt_show_image(image)
        plt.imsave(f"./playground_imgs/diff_steps/diffprocess_sample_{i:02d}.png", image)
        image_reservoir.append(image)

    @torch.no_grad()
    def save_latents(i, t, latents):
        latents_reservoir.append(latents.detach().cpu())

    # Uncoment if you want to visualize diffuzer steps
    # prompt = "a handsome cat dressed like Lincoln, trending art."
    # with torch.no_grad():
    #     image = pipe(prompt, callback=plot_show_callback).images[0]

    # image.save(f"./playground_imgs/lovely_cat_lincoln.png")
    
    # create_video(np.array(image_reservoir), "difuser_steps")

def simple_sampling_fun():
    pass

if __name__ == "__main__":
    simple_generation_diffuser_step_vis()
