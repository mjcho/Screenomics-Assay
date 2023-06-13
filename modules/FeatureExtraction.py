import clip
import os
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import pathlib
from torch import nn
import numpy as np
from torch.utils.data import ChainDataset
import pickle
import argparse
import timm

from datetime import datetime

from PIL import features, Image
from packaging import version

try:
    ver = Image.__version__  # PIL >= 7
except:
    ver = Image.PILLOW_VERSION  # PIL <  7

if version.parse(ver) >= version.parse("5.4.0"):
    if features.check_feature("libjpeg_turbo"):
        print("libjpeg-turbo is on")
    else:
        print("libjpeg-turbo is not on")
else:
    print(
        f"libjpeg-turbo' status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {ver}"
    )

module = "FeatureExtraction"


# Get model helper
def get_model(
    model_name, moco_pth="/home/ezelikman/vit-b-300ep.pth.tar", device="cuda:0"
):
    print(f"Getting model {model_name}")
    if "clip" in model_name:
        models = ["RN50x16", "ViT-B/16", "ViT-L/14"]
        model, _ = clip.load(models[2], device=device)
        encode_image = model.encode_image
    elif "dino_" in model_name:
        model = torch.hub.load("facebookresearch/dino:main", model_name)
        model = torch.nn.DataParallel(model)
        encode_image = model
    elif model_name == "moco":
        import git

        if not os.path.exists("moco-v3"):
            git.Git(".").clone("https://github.com/facebookresearch/moco-v3.git")
        # os.system("git clone https://github.com/facebookresearch/moco-v3.git")
        from mocov3 import vits

        model = vits.vit_base()
        checkpoint = torch.load(moco_pth)
        state_dict = checkpoint["state_dict"]
        linear_keyword = "head"
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith(
                "module.base_encoder.%s" % linear_keyword
            ):
                # remove prefix
                state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.head = nn.Identity()
        encode_image = model
    else:
        model = timm.create_model(model_name, pretrained=True)
        encode_image = lambda x: model.forward_features(x).mean(-1).mean(-1)
    return model.to(device), encode_image


# Pad image for the transform
# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
class SquarePad:
    def __call__(self, image):
        image_shape = image[0].shape
        max_wh = max(image_shape)
        p_left, p_top = [(max_wh - s) // 2 for s in image_shape]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image_shape, [p_left, p_top])
        ]
        padding = (p_top, p_left, p_bottom, p_right)
        return F.pad(image, padding, 0, "constant")


transform = transforms.Compose(
    [
        transforms.Resize(size=224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
        SquarePad(),
        transforms.Resize(size=224),
    ]
)


def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]

    # transform function
    tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[127 / 255, 127 / 255, 127 / 255],
                std=[128 / 255, 128 / 255, 128 / 255],
            ),
        ]
    )

    print(
        f"Initiating {dataset_name} {module} dataset start at: {str(datetime.now())}\n"
    )
    with open(f"{out_dir}/dataset.log", "a", encoding="utf-8") as log:
        log.write(
            f"Initiation {dataset_name} {module} dataset starts at: {str(datetime.now())}\n"
        )
    # if not os.path.exists(image_dataset_file):
    #     dataset = ImageFolder(dirpath, transform=tf)

    dataset = ImageFolder(dirpath, transform=tf)

    print(
        f"Initiation {dataset_name} {module} dataset ends at: {str(datetime.now())}\n"
    )
    with open(f"{out_dir}/dataset.log", "a", encoding="utf-8") as log:
        log.write(
            f"Initiation {dataset_name} {module} dataset ends at: {str(datetime.now())}\n"
        )

    print("Saving dataset...")
    with open(f"{out_dir}/{dataset_name}_dataset_{module}.pkl", "wb") as f:
        pickle.dump(dataset, f, -1)
    print(f"Dataset saved, length = {len(dataset)}\n\n")


def run(dirpath, out_dir, dataset, batch_size, num_workers, device, model):
    dataset_name = dirpath.split("/")[-1]
    print(model)

    # Load model
    print("Loading models...\n\n")
    _, encode_image = get_model(model, device=device)
    print("\n\nModels loaded.\n\n")

    # Load dataset
    print("Loading dataset...")
    print(dataset)
    with open(dataset, "rb") as infile:
        dataset = pickle.load(infile)
    print(f"Dataset loaded, length = {len(dataset)}\n\n")

    # TODO: resumption control
    # is_resume = False

    # Set pipeline params
    # save every save_batch images
    save_batch = 65536
    model_output_dir = f"{out_dir}/{model}"
    # create dir for saving dataset
    if not os.path.isdir(model_output_dir):  # for outputs
        os.mkdir(model_output_dir)

    # Resume from where is left off by checking the saved files
    cur_files = os.listdir(model_output_dir)
    max_idx = max(
        [int(cur_file.split(".")[0].split("_")[-1]) for cur_file in cur_files] + [0]
    )
    if max_idx != 0:
        dataset = torch.utils.data.Subset(dataset, range(max_idx, len(dataset)))
        print(f"Starting from {max_idx} of {len(dataset)}")

    print("Creating dataloader...")
    # Create dataset and loader
    imageloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # each cpu process 30 images per batch
        num_workers=num_workers,
        prefetch_factor=2,
    )
    print(f"dataloader created, length = {len(imageloader)}.\n\n")

    # save batch helper
    def save_batch_array(accumulated_data):
        return np.savez_compressed(
            f"{model_output_dir}/{file_counter * save_batch}_{file_counter * save_batch + len(accumulated_data)}.npz",
            embeddings=accumulated_data,
        )

    """Processing"""
    file_counter = max_idx // save_batch
    accumulated_data = None
    with torch.no_grad():
        for image_idx, (image, _) in enumerate(tqdm(imageloader)):
            image = image.to(device)
            # image = nn.functional.interpolate(image, 224)
            image_encodings = encode_image(image).cpu()
            if accumulated_data is None:
                accumulated_data = [image_encodings.clone()]
            else:
                accumulated_data.append(image_encodings)

            if len(accumulated_data) >= save_batch // batch_size:
                accumulated_data = torch.cat(accumulated_data).contiguous()
                save_batch_array(accumulated_data)
                accumulated_data = None
                file_counter += 1
        if accumulated_data is not None:  # for saving last batches
            accumulated_data = torch.cat(accumulated_data).contiguous()
            save_batch_array(accumulated_data)
            accumulated_data = None
            file_counter += 1
