def load_pipe_into_our_UNet(myUNet, pipe_unet):
    # load the pretrained weights from the pipe into our UNet.
    # Loading input and output layers.
    myUNet.output[0].load_state_dict(pipe_unet.conv_norm_out.state_dict())
    myUNet.output[2].load_state_dict(pipe_unet.conv_out.state_dict())
    myUNet.conv_input.load_state_dict(pipe_unet.conv_in.state_dict())
    myUNet.time_embedding.load_state_dict(pipe_unet.time_embedding.state_dict())
    # % Loading the down blocks
    myUNet.down_blocks[0][0].load_state_dict(
        pipe_unet.down_blocks[0].resnets[0].state_dict()
    )
    myUNet.down_blocks[0][1].load_state_dict(
        pipe_unet.down_blocks[0].attentions[0].state_dict()
    )
    myUNet.down_blocks[1][0].load_state_dict(
        pipe_unet.down_blocks[0].resnets[1].state_dict()
    )
    myUNet.down_blocks[1][1].load_state_dict(
        pipe_unet.down_blocks[0].attentions[1].state_dict()
    )
    myUNet.down_blocks[2][0].load_state_dict(
        pipe_unet.down_blocks[0].downsamplers[0].state_dict()
    )

    myUNet.down_blocks[3][0].load_state_dict(
        pipe_unet.down_blocks[1].resnets[0].state_dict()
    )
    myUNet.down_blocks[3][1].load_state_dict(
        pipe_unet.down_blocks[1].attentions[0].state_dict()
    )
    myUNet.down_blocks[4][0].load_state_dict(
        pipe_unet.down_blocks[1].resnets[1].state_dict()
    )
    myUNet.down_blocks[4][1].load_state_dict(
        pipe_unet.down_blocks[1].attentions[1].state_dict()
    )
    myUNet.down_blocks[5][0].load_state_dict(
        pipe_unet.down_blocks[1].downsamplers[0].state_dict()
    )

    myUNet.down_blocks[6][0].load_state_dict(
        pipe_unet.down_blocks[2].resnets[0].state_dict()
    )
    myUNet.down_blocks[6][1].load_state_dict(
        pipe_unet.down_blocks[2].attentions[0].state_dict()
    )
    myUNet.down_blocks[7][0].load_state_dict(
        pipe_unet.down_blocks[2].resnets[1].state_dict()
    )
    myUNet.down_blocks[7][1].load_state_dict(
        pipe_unet.down_blocks[2].attentions[1].state_dict()
    )
    myUNet.down_blocks[8][0].load_state_dict(
        pipe_unet.down_blocks[2].downsamplers[0].state_dict()
    )

    myUNet.down_blocks[9][0].load_state_dict(
        pipe_unet.down_blocks[3].resnets[0].state_dict()
    )
    myUNet.down_blocks[10][0].load_state_dict(
        pipe_unet.down_blocks[3].resnets[1].state_dict()
    )

    # % Loading the middle blocks
    myUNet.middle_block[0].load_state_dict(pipe_unet.mid_block.resnets[0].state_dict())
    myUNet.middle_block[1].load_state_dict(
        pipe_unet.mid_block.attentions[0].state_dict()
    )
    myUNet.middle_block[2].load_state_dict(pipe_unet.mid_block.resnets[1].state_dict())
    # % Loading the up blocks
    # upblock 0
    myUNet.upsample_output_blocks[0][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[0].state_dict()
    )
    myUNet.upsample_output_blocks[1][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[1].state_dict()
    )
    myUNet.upsample_output_blocks[2][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[2].state_dict()
    )
    myUNet.upsample_output_blocks[2][1].load_state_dict(
        pipe_unet.up_blocks[0].upsamplers[0].state_dict()
    )
    # % upblock 1
    myUNet.upsample_output_blocks[3][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[0].state_dict()
    )
    myUNet.upsample_output_blocks[3][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[0].state_dict()
    )
    myUNet.upsample_output_blocks[4][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[1].state_dict()
    )
    myUNet.upsample_output_blocks[4][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[1].state_dict()
    )
    myUNet.upsample_output_blocks[5][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[2].state_dict()
    )
    myUNet.upsample_output_blocks[5][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[2].state_dict()
    )
    myUNet.upsample_output_blocks[5][2].load_state_dict(
        pipe_unet.up_blocks[1].upsamplers[0].state_dict()
    )
    # % upblock 2
    myUNet.upsample_output_blocks[6][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[0].state_dict()
    )
    myUNet.upsample_output_blocks[6][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[0].state_dict()
    )
    myUNet.upsample_output_blocks[7][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[1].state_dict()
    )
    myUNet.upsample_output_blocks[7][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[1].state_dict()
    )
    myUNet.upsample_output_blocks[8][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[2].state_dict()
    )
    myUNet.upsample_output_blocks[8][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[2].state_dict()
    )
    myUNet.upsample_output_blocks[8][2].load_state_dict(
        pipe_unet.up_blocks[2].upsamplers[0].state_dict()
    )
    # % upblock 3
    myUNet.upsample_output_blocks[9][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[0].state_dict()
    )
    myUNet.upsample_output_blocks[9][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[0].state_dict()
    )
    myUNet.upsample_output_blocks[10][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[1].state_dict()
    )
    myUNet.upsample_output_blocks[10][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[1].state_dict()
    )
    myUNet.upsample_output_blocks[11][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[2].state_dict()
    )
    myUNet.upsample_output_blocks[11][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[2].state_dict()
    )
