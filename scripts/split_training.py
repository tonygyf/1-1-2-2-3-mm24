"""Create reproducible source-group-disjoint training and development JSONs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extremeocr.splits import split_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", help="JSON mapping each source filename to scene/object group")
    args = parser.parse_args()
    groups = json.loads(Path(args.groups).read_text(encoding="utf-8")) if args.groups else None
    print(json.dumps(split_training(args.images, args.annotations, args.output,
                                    args.dev_fraction, args.seed, groups), indent=2))


if __name__ == "__main__":
    main()
