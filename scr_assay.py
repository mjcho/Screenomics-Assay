import argparse
import os
import sys
import time
from datetime import datetime
import pickle
from validate import validate
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    # main parser
    parser = argparse.ArgumentParser(description="Screenomics-assay main method")
    subparsers = parser.add_subparsers(
        title="action to take",
        # description="action to take",
        help="use %(prog)s [action] -h for help for each action",
    )
    # arguments common to all actions (create, run, validate)
    parser.add_argument(
        "dir_path",
        help="the directory of images, first level needs to be classes/participants",
    )
    parser.add_argument(
        "out_dir", help="the directory for output log and invalid images"
    )
    # module_parser, parent of create and run, so both actions share the module argument
    module_parser = argparse.ArgumentParser(description="module", add_help=False)
    module_parser.add_argument(
        "module",
        help="module to run or create dataset for",
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
    )
    # create parser
    create_parser = subparsers.add_parser(
        "create", parents=[module_parser], help="create dataset for a module"
    )
    create_parser.set_defaults(action="create")
    # run parser
    run_parser = subparsers.add_parser(
        "run", parents=[module_parser], help="run module over a dataset"
    )
    run_parser.set_defaults(action="run")
    run_parser.add_argument(
        "-d",
        "--dataset",
        help="the path of the dataset to be run on, default: [out_dir]/[name of the image dir]_dataset_[module].pkl",
    )
    run_parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size of the prediction, default: 32",
        default=32,
    )
    run_parser.add_argument(
        "-nw",
        "--num_workers",
        help="number of workers for the dataloader, default: num of cpu cores",
        default=os.cpu_count(),
    )
    run_parser.add_argument(
        "-dv",
        "--device",
        default="cuda:0",
        help="device to run the model (e.g., 'cpu' or 'cuda:0'), gpu device ordinal is also valid (default is 0)",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        choices=["clip", "dino_vitb8", "moco", "resnet50"],
        default="clip",
        help="model used for feature extraction, default: clip",
    )
    # validate parser
    validate_parser = subparsers.add_parser(
        "validate", help="validate images in a directory"
    )
    validate_parser.set_defaults(action="validate")

    args = parser.parse_args()
    action = args.action
    dir_path = args.dir_path
    out_dir = args.out_dir

    # create out_dir if not exist
    if not os.path.isdir(out_dir):  # for outputs
        os.mkdir(out_dir)

    # execute
    if action == "create":
        # create dir for saving dataset
        create, run = switch_import(args.module)
        create(dir_path, out_dir)
    if action == "run":
        module = args.module
        batch_size = int(args.batch_size)
        num_workers = int(args.num_workers)
        device = args.device
        model = args.model

        # Check device, if device is cpu, keep it as cpu. If device is cuda:[gpu_ind], keep it as cuda:[gpu_ind].
        # If device is int, make it cuda:[int]. if device is not any of the above, raise error.
        if device == "cpu":
            pass
        elif device.startswith("cuda:"):
            if device[5:].isdigit():
                pass
            else:
                raise ValueError("device must be cuda:[gpu_id] an int or 'cpu'")
        elif device.isdigit():
            device = f"cuda:{device}"
        else:
            raise ValueError("device must be an int or 'cpu'")

        # create dataset if not provided
        if not args.dataset:
            dataset = f"{out_dir}/{dir_path.split('/')[-1]}_dataset_{module}.pkl"
        else:
            dataset = args.dataset

        # run module
        create, run = switch_import(module)
        if module != "FeatureExtraction":
            run(dir_path, out_dir, dataset, batch_size, num_workers, device)
        else:
            run(dir_path, out_dir, dataset, batch_size, num_workers, device, model)
    if action == "validate":
        sys.argv = [sys.argv[0], dir_path, "--out_dir", out_dir]
        validate(sys.argv)
