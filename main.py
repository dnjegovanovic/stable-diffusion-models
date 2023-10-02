from argparse import ArgumentParser

import playground.stable_diffusiuon_playground as pg


def main(args):
    if args.test_playground:
        pg.main(args)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--test-playground", action="store_true", help="Simple test")

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
    
    main(parser.parse_args())
