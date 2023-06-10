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
import easyocr
import cv2

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
module = "OCR"


""" Create dataset """


def img_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]

    new_h, new_w = 640, 360  # 560, 315 # 800, 450  480, 270

    tf = transforms.Compose(
        [
            transforms.Resize((new_h, new_w)),
            # transforms.ToTensor()
        ]
    )

    print(
        f"Initiating {dataset_name} {module} dataset start at: {str(datetime.now())}\n"
    )
    with open(f"{out_dir}/dataset.log", "a", encoding="utf-8") as log:
        log.write(
            f"Initiation {dataset_name} {module} dataset starts at: {str(datetime.now())}\n"
        )
    dataset = torchvision.datasets.ImageFolder(dirpath, transform=tf, loader=img_loader)

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
    # Initiate reader
    reader = easyocr.Reader(["en"], model_storage_directory="./easyocr", gpu=device)

    # update reader.readtext_batched params so there's grey images are processed at the dataloader level (save some time)
    # ref. https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py
    import types
    from easyocr.utils import reformat_input_batched

    def readtext_batched(
        self,
        img,
        img_cv_grey,
        n_width=None,
        n_height=None,
        decoder="greedy",
        beamWidth=5,
        batch_size=1,
        workers=0,
        allowlist=None,
        blocklist=None,
        detail=1,
        rotation_info=None,
        paragraph=False,
        min_size=20,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        y_ths=0.5,
        x_ths=1.0,
        add_margin=0.1,
        output_format="standard",
    ):
        """
        Parameters:
        image: file path or numpy-array or a byte stream object
        When sending a list of images, they all must of the same size,
        the following parameters will automatically resize if they are not None
        n_width: int, new width
        n_height: int, new height
        """
        # img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

        horizontal_list_agg, free_list_agg = self.detect(
            img,
            min_size,
            text_threshold,
            low_text,
            link_threshold,
            canvas_size,
            mag_ratio,
            slope_ths,
            ycenter_ths,
            height_ths,
            width_ths,
            add_margin,
            False,
        )
        result_agg = []
        # put img_cv_grey in a list if its a single img
        img_cv_grey = [img_cv_grey] if len(img_cv_grey.shape) == 2 else img_cv_grey
        for grey_img, horizontal_list, free_list in zip(
            img_cv_grey, horizontal_list_agg, free_list_agg
        ):
            result_agg.append(
                self.recognize(
                    grey_img,
                    horizontal_list,
                    free_list,
                    decoder,
                    beamWidth,
                    batch_size,
                    workers,
                    allowlist,
                    blocklist,
                    detail,
                    rotation_info,
                    paragraph,
                    contrast_ths,
                    adjust_contrast,
                    filter_ths,
                    y_ths,
                    x_ths,
                    False,
                    output_format,
                )
            )
        return result_agg

    reader.readtext_batched = types.MethodType(readtext_batched, reader)
    print("\n\nModels loaded.\n\n")

    # Load dataset
    print("Loading dataset...")
    print(dataset)

    def img_loader(path: str):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    with open(dataset, "rb") as infile:
        dataset = pickle.load(infile)

    print(f"Dataset loaded, length = {len(dataset)}\n\n")

    print("Creating dataloader...")

    # Create dataset and loader
    def pil_collate_fn(batch):
        # batch is a list of tuples (PIL.Image, class) from ImageFolder
        # returns [(PIL.Image0, Image1,...), (class0, class1)]
        # ref. is default_collate in https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py

        # # if img.size is needed
        # batch = [(b[0], b[0].size, b[1]) for b in batch]

        # # if convert to gray at the dataloader time
        # # note np.array(b[0]) is RGB. Need to change package code. See "Important notes".
        batch = [
            (np.array(b[0]), cv2.cvtColor(np.array(b[0]), cv2.COLOR_RGB2GRAY), b[1])
            for b in batch
        ]
        return list(zip(*batch))

    imageloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # each cpu process 30 images per batch
        num_workers=num_workers,
        prefetch_factor=2,
        collate_fn=pil_collate_fn,
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

    for i, (inputs, inputs_grey, _) in tqdm(enumerate(imageloader)):
        # (if resume for stop) actual batch no. is i + last_batch + 1
        # if is_resume:
        #     i = i + last_batch

        # create batch entry in error log
        err[i] = {"input": "", "inference": "", "output": ""}

        """Print and log progress"""
        t_bstart = time.time()
        if (i + 1) % 100 == 0:
            batch_log = f"Doing batch {i + 1}/{n_batch}, time elapsed: {int((t_bstart-t0)/60)}m {round((t_bstart-t0)%60, 2)}s. \n"
            print(batch_log)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(batch_log)

        """Convert input array"""
        try:
            inputs = np.stack(inputs)
            inputs_grey = np.stack(inputs_grey)

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
                res = reader.readtext_batched(inputs, inputs_grey, batch_size=300)

        except Exception as e:
            t_inf = time.time()
            inf_log = f"\t\t Failed inferencing for batch {i + 1}, time elapsed: {int((t_inf-t0)/60)}m {round((t_inf-t0)%60, 2)}s. \n"
            print(inf_log)
            err[i]["inference"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(inf_log)

        """Save output"""
        try:
            # actual_i = (proc_id * subset_len) + (i * batch_size)
            actual_i = i * batch_size
            if is_resume:
                if i != n_batch - 1:
                    img_names = [
                        img_name.split("/")[-1]
                        for (
                            img_name,
                            _,
                        ) in imageloader.dataset.dataset.samples[
                            (actual_i) : (actual_i + batch_size)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (
                            img_name,
                            _,
                        ) in imageloader.dataset.dataset.samples[
                            (actual_i) : (actual_i + len(inputs))
                        ]
                    ]
            else:
                if i != n_batch - 1:
                    img_names = [
                        img_name.split("/")[-1]
                        for (
                            img_name,
                            _,
                        ) in imageloader.dataset.samples[
                            (actual_i) : (actual_i + batch_size)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (
                            img_name,
                            _,
                        ) in imageloader.dataset.samples[
                            (actual_i) : (actual_i + len(inputs))
                        ]
                    ]

            # save output
            output_name = f"{i}.{img_names[0]}.{img_names[-1]}.npz"
            np.savez(
                model_res_dir + "/" + output_name,
                result=np.asarray(res, dtype=object),
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
    done_log = f"Done all {n_batch} batches at: {str(datetime.now())}.\n Elapsed time: {int((t_done-t0)/60)}m {round((t_done-t0)%60, 2)}s. \n"
    print(done_log)
    with open(model_log_path, "a", encoding="utf-8") as ap_log:
        ap_log.write(done_log)
