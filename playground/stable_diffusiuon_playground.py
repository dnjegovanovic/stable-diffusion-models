import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

import json
with open('./token.json', 'r') as f:
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
        data = pipe(prompt, generator=generator, num_inference_steps=num_inference_steps)
    else:
        data = pipe(prompt, generator=generator)
    image = data.images[0]
    image.save(f"./playground_imgs/{img_name}.png")
    plt_show_image(image)


def main():
    pass


if __name__ == "__main__":
    generator = torch.Generator("cuda").manual_seed(1024)

    prompt1 = "a lovely cat running in the desert in Van Gogh style, trending art."
    prompt2 = "a sleeping cat enjoying the sunshine."
    prompt1 = "a lovely cat running in the desert in Van Gogh style, trending art."
    pipe = create_pipe()
    create_simple_img(pipe, prompt2, "nfprompt_sunshine_interference_cat", generator)
