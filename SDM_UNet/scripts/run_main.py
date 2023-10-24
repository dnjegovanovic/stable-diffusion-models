import torch

from SDM_UNet.modules.UNetStableDiffusin import *
from SDM_UNet.scripts.load_weights import load_pipe_into_our_UNet
from diffusers import StableDiffusionPipeline
import json

with open("./token.json", "r") as f:
    data_token = json.load(f)

from huggingface_hub import login

login(data_token["token"])


def generate_image(prompt):

    device = "cuda"
    model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=True)
    pipe = pipe.to(device)

    myunet = UNetStableDiffusion()
    original_unet = pipe.unet.cpu()
    load_pipe_into_our_UNet(myunet, original_unet)

    pipe.unet = myunet.cuda()
    # prompt = "A ballerina riding a Harley Motorcycle, CG Art"
    with torch.no_grad():
        image = pipe(prompt).images[0]

    image.save("./playground_imgs/SDM_UNet/test.png")
