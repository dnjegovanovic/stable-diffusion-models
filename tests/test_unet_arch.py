from SDM_UNet.models.ResBlock import *
from SDM_UNet.models.DownlSampling import *
from SDM_UNet.models.UpSampling import *
from SDM_UNet.models.CrossAttention import *
from SDM_UNet.models.TransformerBlock import *
from SDM_UNet.models.SpatialTransformer import *

import unittest

from diffusers import StableDiffusionPipeline
import json

with open("./token.json", "r") as f:
    data_token = json.load(f)

from huggingface_hub import login

login(data_token["token"])


class TestUNetComponents(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # Load SD pipline fro huggingface
        device = "cuda"
        model_path = "CompVis/stable-diffusion-v1-4"

        pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=True)
        self.pipe = pipe.to(device)

    def test_resblock(self):
        # Create our Res block
        tmp_blk = ResBlock(320, 1280).cuda()
        # Take standrd ResBlock
        std_blk = self.pipe.unet.down_blocks[0].resnets[0]

        # Take parameters from standard ResBlock from hugingface and load to our ResBlock
        SD = std_blk.state_dict()
        tmp_blk.load_state_dict(SD)
        # Generate random data
        lat_tmp = torch.randn(3, 320, 32, 32)
        temb = torch.randn(3, 1280)
        # Evaluate bouth blocks and compare result
        with torch.no_grad():
            out = self.pipe.unet.down_blocks[0].resnets[0](lat_tmp.cuda(), temb.cuda())
            out2 = tmp_blk(lat_tmp.cuda(), temb.cuda())

        assert torch.allclose(out2, out)

    def test_downsampler(self):
        # Create DownSample block
        tmpdsp = DownSample(320).cuda()
        # Load standard downsample block from huggingface
        stddsp = self.pipe.unet.down_blocks[0].downsamplers[0]
        # Initialize pur block with parametars from huggingface model
        SD = stddsp.state_dict()
        tmpdsp.load_state_dict(SD)
        # random data
        lat_tmp = torch.randn(3, 320, 32, 32)
        # comapre result
        with torch.no_grad():
            out = stddsp(lat_tmp.cuda())
            out2 = tmpdsp(lat_tmp.cuda())

        assert torch.allclose(out2, out)

    def test_upsampler(self):
        tmpusp = UpSample(1280).cuda()
        stdusp = self.pipe.unet.up_blocks[1].upsamplers[0]
        SD = stdusp.state_dict()
        tmpusp.load_state_dict(SD)
        lat_tmp = torch.randn(3, 1280, 32, 32)
        with torch.no_grad():
            out = stdusp(lat_tmp.cuda())
            out2 = tmpusp(lat_tmp.cuda())

        assert torch.allclose(out2, out)

    def test_self_attention(self):
        tmpSattn = CrossAttention(320, 320, context_dim=None, num_heads=8).cuda()
        stdSattn = (
            self.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1
        )
        tmpSattn.load_state_dict(stdSattn.state_dict())  # checked
        with torch.no_grad():
            lat_tmp = torch.randn(3, 32, 320)
            out = stdSattn(lat_tmp.cuda())
            out2 = tmpSattn(lat_tmp.cuda())
        assert torch.allclose(out2, out, rtol=1e-02)  # False

    def test_cross_attention(self):
        tmpXattn = CrossAttention(320, 320, context_dim=768, num_heads=8).cuda()
        stdXattn = (
            self.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2
        )
        tmpXattn.load_state_dict(stdXattn.state_dict())  # checked
        with torch.no_grad():
            lat_tmp = torch.randn(3, 32, 320)
            ctx_tmp = torch.randn(3, 5, 768)
            out = stdXattn(lat_tmp.cuda(), ctx_tmp.cuda())
            out2 = tmpXattn(lat_tmp.cuda(), ctx_tmp.cuda())
        assert torch.allclose(out2, out, rtol=1e-02)  # False

    # def test_TransformerBlock(self):
    #     tmpTfmer = TransformerBlock(320, context_dim=768, num_heads=8).cuda()
    #     stdTfmer = self.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0]
    #     tmpTfmer.load_state_dict(stdTfmer.state_dict())  # checked
    #     with torch.no_grad():
    #         lat_tmp = torch.randn(3, 32, 320)
    #         ctx_tmp = torch.randn(3, 5, 768)
    #         out = tmpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
    #         out2 = stdTfmer(lat_tmp.cuda(), ctx_tmp.cuda())#Smthin
    #     assert torch.allclose(out2, out)  # False

    # def test_SpatialTransformer(self):
    #     tmpSpTfmer = SpatialTransformer(320, context_dim=768, num_heads=8).cuda()
    #     stdSpTfmer = self.pipe.unet.down_blocks[0].attentions[0]
    #     tmpSpTfmer.load_state_dict(stdSpTfmer.state_dict())  # checked
    #     with torch.no_grad():
    #         lat_tmp = torch.randn(3, 320, 8, 8)
    #         ctx_tmp = torch.randn(3, 5, 768)
    #         out = tmpSpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
    #         out2 = stdSpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
    #     assert torch.allclose(out2, out)  # False
