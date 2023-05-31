import argparse
import os
import sys
import time
from datetime import datetime
import pickle
from validate import validate


def switch_import(module):
    if module == "DeepAffect":
        from modules.DeepAffect import create, run
    if module == "Face":
        from modules.Face import create, run
    if module == "Food":
        from modules.Food import create, run
    if module == "ImageClassification":
        from modules.ImageClassification import create, run
    if module == "ImageProperties":
        from modules.ImageProperties import create, run
    if module == "ObjectDetection":
        from modules.ObjectDetection import create, run
    if module == "OCR":
        from modules.OCR import create, run
    if module == "Places365":
        from modules.Places365 import create, run
    if module == "FeatureExtraction":
        from modules.FeatureExtraction import create, run

    return create, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset for or run a module")
    parser.add_argument(
        "action",
        choices=["create", "run", "validate"],
        help="create dataset or run module?",
    )
    parser.add_argument(
        "module",
        choices=[
            "DeepAffect",
            "Face",
            "Food",
            "ImageClassification",
            "ImageProperties",
            "ObjectDetection",
            "OCR",
            "Places365",
            "FeatureExtraction",
        ],
        help="module name",
    )
    parser.add_argument(
        "dirpath",
        help="the directory of the images (relative to user dir), first level needs to be classes/participants",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="the directory for output log and invalid images (relative to user dir), default is user dir",
        default=os.path.expanduser("~"),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="the path of the dataset to be run on (relative to user dir)",
        # default=os.path.expanduser(f"{dirpath.split('/')}_dataset_{MODULE}.pkl"),
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size of the prediction",
        # default=os.path.expanduser(f"{dirpath.split('/')}_dataset_{MODULE}.pkl"),
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["moco", "dino_ditv8", "resnet50"],
        help="model used for feature extraction",
    )

    args = parser.parse_args()
    action = args.action
    module = args.module
    dirpath = os.path.expanduser(f"~/{args.dirpath}")
    out_dir = os.path.expanduser(f"~/{args.out_dir}")
    if not args.dataset:
        args.dataset = f"{dirpath.split('/')[-1]}_dataset_{module}.pkl"

    dataset = os.path.expanduser(f"~/{args.dataset}")
    batch_size = args.batch_size
    model = args.model
    # print(action, module, dirpath, out_dir, dataset)

    # import module
    create, run = switch_import(module)

    # execute
    if action == "create":
        create(dirpath, out_dir)
    if action == "run":
        if batch_size:
            if module != "FeatureExtraction":
                run(dirpath, out_dir, dataset, batch_size)
            else:
                run(dirpath, out_dir, dataset, batch_size, model)
        else:
            if module != "FeatureExtraction":
                run(dirpath, out_dir, dataset, 32)
            else:
                run(dirpath, out_dir, dataset, 32, model)
    if action == "validate":
        sys.argv = [sys.argv[0], dirpath, "--out_dir", out_dir]
        validate(sys.argv)
