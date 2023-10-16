from SDM_UNet.models.ResBlock import *
from diffusers import StableDiffusionPipeline
import json

# Load SD pipline fro huggingface
device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=True)
pipe = pipe.to(device)


def test_resblock():
    # Create our Res block
    tmp_blk = ResBlock(320, 1280).cuda()
    # Take standrd ResBlock
    std_blk = pipe.unet.down_blocks[0].resnets[0]

    # Take parameters from standard ResBlock from hugingface and load to our ResBlock
    SD = std_blk.state_dict()
    tmp_blk.load_state_dict(SD)
    # Generate random data
    lat_tmp = torch.randn(3, 320, 32, 32)
    temb = torch.randn(3, 1280)
    # Evaluate bouth blocks and compare result
    with torch.no_grad():
        out = pipe.unet.down_blocks[0].resnets[0](lat_tmp.cuda(), temb.cuda())
        out2 = tmp_blk(lat_tmp.cuda(), temb.cuda())

    assert torch.allclose(out2, out)
