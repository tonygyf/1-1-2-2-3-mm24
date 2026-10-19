"""Score a frozen recognizer's crop predictions; reject missing/duplicate ids."""
import argparse
import json
from pathlib import Path


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, char in enumerate(first, 1):
        current = [i]
        for j, other in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def score(instances, predictions):
    def index(rows):
        output = {}
        for row in rows:
            key = (row["image_id"], row["annotation_id"])
            if key in output:
                raise ValueError("Duplicate crop identity")
            output[key] = row
        return output
    truth, predicted = index(instances), index(predictions)
    if truth.keys() != predicted.keys() or not truth:
        raise ValueError("Predictions must exactly cover the nonempty crop set")
    correct, errors, characters = 0, 0, 0
    for key, row in truth.items():
        target, text = row["text"].casefold().strip(), predicted[key]["text"].casefold().strip()
        if not target:
            raise ValueError("Empty GT transcription requires explicit review")
        correct += target == text
        errors += edit_distance(target, text)
        characters += len(target)
    return {"instances": len(truth), "word_accuracy": correct / len(truth),
            "cer": errors / characters, "normalization": "Unicode casefold + outer strip",
            "protocol": "GT-axis-aligned-crop-diagnostic; not official e2e"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(score(read(args.instances), read(args.predictions)), indent=2))


if __name__ == "__main__":
    main()
