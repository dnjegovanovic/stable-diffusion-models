from argparse import ArgumentParser

import playground.stable_diffusiuon_playground as pg

from SDM_UNet.scripts.run_main import *
import SDM_Pipeline_MNIST.scripts.test_train_simpleUNet as sunet
import SDM_Pipeline_MNIST.scripts.test_train_UNetTransformer as unettr
from SDM_Pipeline_MNIST.config.core import config


def main(args):
    if args.test_playground:
        pg.main(args)
    elif args.test_custom_unet:
        generate_image(args.prompt)
    elif args.test_minst_unet:
        sunet.main(config)
    elif args.test_minst_unettr:
        unettr.main(config)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--test-playground", action="store_true", help="Simple test")
    parser.add_argument("--test-custom-unet", action="store_true", help="Simple test")
    parser.add_argument(
        "--test-minst-unet", action="store_true", help="test-minst-unet"
    )
    parser.add_argument(
        "--test-minst-unettr", action="store_true", help="test unet transformer model"
    )
    # ---------------------------------------------------------------------
    playground_parser = parser.add_argument_group("Playground arguments")
    playground_parser.add_argument(
        "--simple-gen", action="store_true", help="Simple test"
    )
    playground_parser.add_argument(
        "--debug-simple-sampling", action="store_true", help="path to dataset"
    )
    playground_parser.add_argument(
        "--image-to-image",
        action="store_true",
        help="Test image to image process",
    )
    # ---------------------------------------------------------------------
    playground_parser = parser.add_argument_group("Custom UNet arguments")
    playground_parser.add_argument(
        "--prompt",
        type=str,
        default=r"A ballerina riding a Harley Motorcycle, CG Art",
        help="Type prompt to test.",
    )
    main(parser.parse_args())
