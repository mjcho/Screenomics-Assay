import argparse
import os
import time
from datetime import datetime
import pickle


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

    return create, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset for or run a module")
    parser.add_argument(
        "action",
        choices=["create", "run"],
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

    args = parser.parse_args()
    action = args.action
    module = args.module
    dirpath = os.path.expanduser(f"~/{args.dirpath}")
    out_dir = os.path.expanduser(f"~/{args.out_dir}")

    # import module
    create, run = switch_import(module)

    if action == "create":
        create(dirpath, out_dir)
    else:
        run(dirpath, out_dir)
