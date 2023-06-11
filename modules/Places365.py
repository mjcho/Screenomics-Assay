import torch
from torch.autograd import Variable as V
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2
import time
import json
import pickle
from datetime import datetime
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
module = "Places365"


""" Create dataset """


def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_labels():
    # prepare all the labels

    # scene category labels
    file_name_category = "categories_places365.txt"
    if not os.access(file_name_category, os.W_OK):
        synset_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
        os.system("wget " + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])
    classes = tuple(classes)

    # indoor and outdoor labels
    file_name_IO = "IO_places365.txt"
    if not os.access(file_name_IO, os.W_OK):
        synset_url = "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt"
        os.system("wget " + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute labels
    file_name_attribute = "labels_sunattribute.txt"
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = "https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt"
        os.system("wget " + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = "W_sceneattribute_wideresnet18.npy"
    if not os.access(file_name_W, os.W_OK):
        synset_url = "http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy"
        os.system("wget " + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


# def hook_feature(module, input, output):
#     features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnTF():
    # load the image transformer
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = "wideresnet18_places365.pth.tar"
    if not os.access(model_file, os.W_OK):
        os.system("wget http://places2.csail.mit.edu/models_places365/" + model_file)
        os.system(
            "wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py"
        )

    import wideresnet

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    # hook the feature extractor
    # features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    # for name in features_names:
    #     model._modules.get(name).register_forward_hook(hook_feature)

    # no need to hook 'layer4' because it's not used when computing scene attributes
    # model._modules.get('avgpool').register_forward_hook(hook_feature)

    return model


def run(dirpath, out_dir, dataset, batch_size, num_workers, device):
    dataset_name = dirpath.split("/")[-1]

    # Load model
    print("Loading models...\n\n")
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the model
    model = load_model()
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

            def hook_feature(module, input, output):
                features_blobs.append(np.squeeze(output.data.cpu().numpy()))

            handle = model._modules.get("avgpool").register_forward_hook(hook_feature)
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
            with torch.no_grad():  # save some gpu memory
                features_blobs = []
                logit = model.forward(inputs)
                handle.remove()

            logit = logit.detach().cpu().numpy()
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
                model_res_dir + "/" + output_name,
                logit=logit,
                features=features_blobs[0],
                img_names=img_names,
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
    done_log = f"Done all {n_batch} batches at: {str(datetime.now())}\n. Elapsed time: {int((t_done-t0)/60)}m {round((t_done-t0)%60, 2)}s. \n"
    print(done_log)
    with open(model_log_path, "a", encoding="utf-8") as ap_log:
        ap_log.write(done_log)
