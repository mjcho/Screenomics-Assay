import torch
import torchvision
from torchvision import transforms
import time
import numpy as np
from datetime import datetime
import pickle
import json
from tqdm import tqdm
import os

from PIL import features, Image
from packaging import version

try:
    ver = Image.__version__  # PIL >= 7
except:
    ver = Image.PILLOW_VERSION  # PIL <  7

if version.parse(ver) >= version.parse("5.4.0"):
    if features.check_feature("libjpeg_turbo"):
        print("\n\nlibjpeg-turbo is on\n\n")
    else:
        print("\n\nlibjpeg-turbo is not on\n\n")
else:
    print(
        f"\n\nlibjpeg-turbo' status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {ver}\n\n"
    )


# Set params
module = "ImageProperties"

""" Create dataset """


def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]

    new_size = 64
    tf = transforms.Compose(
        [
            transforms.Resize(
                (new_size, new_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    print(
        f"Initiating {dataset_name} {module} dataset start at: {str(datetime.now())}\n"
    )
    with open(f"{out_dir}/dataset.log", "a", encoding="utf-8") as log:
        log.write(
            f"Initiation {dataset_name} {module} dataset starts at: {str(datetime.now())}\n"
        )

    dataset = torchvision.datasets.ImageFolder(dirpath, transform=tf)

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


""" Run module"""


def run(dirpath, out_dir, dataset, batch_size, num_workers, device):
    dataset_name = dirpath.split("/")[-1]

    # Load model
    print("Loading models...\n\n")
    if not os.path.exists("Screenomics-Assay/image-entropy-GPU"):
        print("Cloning image_entropy_GPU...")
        import git

        git.Git("Screenomics-Assay").clone(
            "https://github.com/lusinlu/image-entropy-GPU.git"
        )
    import importlib

    entropy = importlib.import_module("image-entropy-GPU.entropy")

    # set patch_size to new_size so no patching is performed (compute for the whole image)
    new_size = 64
    model = entropy.Entropy(
        patch_size=new_size, image_width=new_size, image_height=new_size
    )
    model.eval()
    model.to(device)
    print("\n\nModels loaded.\n\n")

    # Load dataset
    print("Loading dataset...")
    print(dataset)
    with open(dataset, "rb") as infile:
        dataset = pickle.load(infile)
    print(f"Dataset loaded, length = {len(dataset)}\n\n")

    print("Creating dataloader...")
    # Create dataset and loader
    imageloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # each cpu process 30 images per batch
        num_workers=num_workers,
        prefetch_factor=2,
    )
    print(f"dataloader created, length = {len(imageloader)}.\n\n")

    # TODO: resumption control
    is_resume = False

    # Set pipeline params
    # out_dir = f"./assay_output_{dataset_name}"
    model_out_dir = out_dir + f"/{module}"
    model_res_dir = model_out_dir + f"/results"
    model_log_dir = model_out_dir + "/logs"
    model_log_path = model_log_dir + f"/output_log.txt"
    model_err_path = model_log_dir + f"/err_proc.json"
    n_batch = len(imageloader)
    # n_batch = len(imageloader) + last_batch if is_resume else len(imageloader)

    # create output dirs
    # structure: assay_output---ImageClassification---results
    #                        ---OtherModules..     ---logs
    # if not os.path.isdir(out_dir):  # for outputs
    #     os.mkdir(out_dir)
    if not os.path.isdir(model_out_dir):  # for a specific model
        os.mkdir(model_out_dir)
    if not os.path.isdir(model_res_dir):  # for model outputs
        os.mkdir(model_res_dir)
    if not os.path.isdir(model_log_dir):  # for pipeline and error logs
        os.mkdir(model_log_dir)

    """Processing start"""
    t0 = time.time()
    print(f"Processing starts at: {str(datetime.now())}\n")
    with open(model_log_path, "a", encoding="utf-8") as ap_log:
        ap_log.write(f"Processing starts at: {str(datetime.now())}\n")

    # for logging error messages
    err = {}

    for i, (inputs, labels) in enumerate(tqdm(imageloader)):
        # actual batch no. is i + last_batch + 1
        # if is_resume:
        #     i = i + last_batch

        # create batch entry in error log
        err[i] = {"input": "", "inference": "", "output": ""}

        """Print and log progress"""
        t_bstart = time.time()
        if (i + 1) % 1000 == 0:
            batch_log = f"Doing batch {i + 1}/{n_batch}, time elapsed: {int((t_bstart-t0)/60)}m {round((t_bstart-t0)%60, 2)}s. \n"
            print(batch_log)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(batch_log)

        """Convert input array"""
        try:
            inputs = inputs.to(device)

        except Exception as e:
            t_input = time.time()
            input_log = f"\t\t Failed converting input array for batch {i + 1}, time elapsed: {int((t_input-t0)/60)}m {round((t_input-t0)%60, 2)}s. \n"
            print(input_log)
            err[i]["input"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(input_log)

        """Inference"""
        try:
            with torch.no_grad():
                ent_vals = model(inputs)
                ent_vals = ent_vals.detach().cpu().numpy()  # shape = (batch_size, 1)
                ent_vals = ent_vals.squeeze(1)  # shape = (batch_size, )

        except Exception as e:
            t_inf = time.time()
            inf_log = f"\t\t Failed inferencing for batch {i + 1}, time elapsed: {int((t_inf-t0)/60)}m {round((t_inf-t0)%60, 2)}s. \n"
            print(inf_log)
            err[i]["inference"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(inf_log)

        """Save output"""
        try:
            # get image names from the loader based on batch size
            if (
                is_resume
            ):  # location of image names is different when using a subset to initiate dataloader
                if i != n_batch - 1:
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.dataset.samples[
                            (i * batch_size) : (i * batch_size + 360)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.dataset.samples[
                            (i * batch_size) :
                        ]
                    ]
            else:
                if i != n_batch - 1:
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.samples[
                            (i * batch_size) : (i * batch_size + 360)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.samples[
                            (i * batch_size) :
                        ]
                    ]

            # save output
            output_name = f"{i}.{img_names[0]}.{img_names[-1]}.npz"
            np.savez(
                model_res_dir + "/" + output_name, entropy=ent_vals, img_names=img_names
            )

        except Exception as e:
            t_output = time.time()
            output_log = f"\t\t Failed saving output for batch {i + 1}, time elapsed: {int((t_output-t0)/60)}m {round((t_output-t0)%60, 2)}s. \n"
            print(output_log)
            err[i]["output"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(output_log)

        # delete error entry if no error occurred for this batch
        if all(v == "" for v in err[i].values()):
            del err[i]

    # save error as json file
    if err:
        with open(model_err_path, "w", encoding="utf-8") as ef:
            json.dump(err, ef, indent=4)

    t_done = time.time()
    done_log = f"Done all {n_batch} batches at: {str(datetime.now())}. Elapsed time: {int((t_done-t0)/60)}m {round((t_done-t0)%60, 2)}s. \n"
    print(done_log)
    with open(model_log_path, "a", encoding="utf-8") as ap_log:
        ap_log.write(done_log)
