import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
import mediapy as media
import json
import numpy as np
from argparse import ArgumentParser

import requests
from io import BytesIO
from PIL import Image
6

from fun import *
from misc import *

with open("./token.json", "r") as f:
    data_token = json.load(f)

from huggingface_hub import login

login(data_token["token"])


def image_to_image_test(print_model_shape=True):
    device = "cuda"
    model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path, revision="fp16", torch_dtype=torch.float16, use_auth_token=True
    )
    pipe = pipe.to(device)
    
    if print_model_shape:
        # recursive_print(pipe.text_encoder, deepest=3)
        # recursive_print(pipe.text_encoder.text_model.encoder, deepest=3)
        # recursive_print(pipe.text_encoder.text_model.encoder.layers[0], deepest=3)
        # recursive_print(pipe.unet, deepest=2)
        # recursive_print(pipe.unet.down_blocks[0], deepest=2)
        # recursive_print(pipe.unet.up_blocks[2], deepest=2)
        recursive_print(pipe.vae, deepest=3)

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_img = Image.open(BytesIO(response.content)).convert("RGB")
    init_img = init_img.resize((768, 512))

    prompt = "A fantasy landscape, trending on artstation"
    generator = torch.Generator(device=device).manual_seed(1024)
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            image=init_img,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        plt_save_image(image, "./playground_imgs/img_to_img_01.png")


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
    shape = (512, 512)
    output_path = f"./playground_imgs/{video_name}.mp4"
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
        plt.imsave(
            f"./playground_imgs/diff_steps/diffprocess_sample_{i:02d}.png", image
        )
        image_reservoir.append(image)

    @torch.no_grad()
    def save_latents(i, t, latents):
        latents_reservoir.append(latents.detach().cpu())

    # Uncomment if you want to visualize diffuzer steps
    prompt = "a handsome cat dressed like Lincoln, trending art."
    with torch.no_grad():
        image = pipe(prompt, callback=plot_show_callback).images[0]

    image.save(f"./playground_imgs/lovely_cat_lincoln.png")

    create_video(np.array(image_reservoir), "difuser_steps")


def debug_sampling_fun():
    pipe = create_pipe()
    img = simple_sampling_fun(pipe)
    plt_save_image(img, "./playground_imgs/debug-sampling-fun.png")


def main(args):
    if args.simple_gen:
        simple_generation_diffuser_step_vis()
    elif args.debug_simple_sampling:
        debug_sampling_fun()
    elif args.image_to_image:
        image_to_image_test()


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--simple-gen", action="store_true", help="Simple test")
    parser.add_argument(
        "--debug-simple-sampling", action="store_true", help="path to dataset"
    )
    parser.add_argument(
        "--image-to-image",
        action="store_true",
        help="Test image to image process",
    )
    args = parser.parse_args()
    main(args)
