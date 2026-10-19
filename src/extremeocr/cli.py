import argparse
import json

from .data import build_suite


def main():
    parser = argparse.ArgumentParser(description="ExtremeOCR research harness")
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("build", help="Build a paired corruption suite from COCO-style labels")
    build.add_argument("--images", required=True)
    build.add_argument("--annotations", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--config", default="configs/screen.json")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_suite(args.images, args.annotations, args.output, args.config)))


if __name__ == "__main__":
    main()
