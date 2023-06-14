import os
import torch
import torchvision
from torchvision import transforms
import imghdr
import time
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import types
import json

from facelib import AgeGenderEstimator, EmotionDetector
import pickle
from datetime import datetime

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
module = "Face"


""" Create dataset """


def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]

    # transform function
    tf = transforms.Compose(
        [
            transforms.Resize((640, 360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 255, 1 / 255, 1 / 255]),
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
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        select_largest=True,
        keep_all=True,
        device=device,
    )
    age_gender_detector = AgeGenderEstimator(device=device)
    emotion_detector = EmotionDetector(device=device)
    print("\n\nModels loaded.\n\n")

    # Forward function for MTCNN to return results for analysis
    def forward(self, img, save_path=None, return_prob=True):
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes,
                batch_probs,
                batch_points,
                img,
                method=self.selection_method,
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:
            return faces, batch_boxes, batch_probs, batch_points
        else:
            return faces

    mtcnn.forward = types.MethodType(forward, mtcnn)
    mtcnn = mtcnn.eval()

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
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
    )
    print(f"dataloader created, length = {len(imageloader)}.\n\n")

    # TODO: rusumption control
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
    # ---OtherModules..     ---logs
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

    for i, (imgs, labels) in enumerate(tqdm(imageloader)):
        # actual batch no. is i + last_batch + 1
        # if is_resume:
        #     i = i + last_batch

        # create batch entry in error log
        # err[i] = {'input': '', 'inference': '', 'analysis': '', 'output': ''}
        err[i] = {}

        """Print and log progress"""
        t_bstart = time.time()
        if (i + 1) % 1000 == 0:
            batch_log = f"Doing batch {i + 1}/{n_batch}, time elapsed: {int((t_bstart-t0)/60)}m {round((t_bstart-t0)%60, 2)}s. \n"
            print(batch_log)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(batch_log)

        """Convert input array"""
        try:
            imgs = imgs.to(device).permute(0, 2, 3, 1)

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
                faces, batch_boxes, batch_probs, batch_points = mtcnn(imgs)

        except Exception as e:
            t_inf = time.time()
            inf_log = f"\t\t Failed inferencing for batch {i + 1}, time elapsed: {int((t_inf-t0)/60)}m {round((t_inf-t0)%60, 2)}s. \n"
            print(inf_log)
            err[i]["inference"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(inf_log)

        """Analysis"""
        try:
            # Save if not face detected
            if all(f is None for f in faces):
                t_save = time.time()
                age = []
                gender = []
                emotion = []
                emotion_prob = []
                if is_resume:
                    if i != n_batch - 1:
                        img_names = [
                            img_name.split("/")[-1]
                            for (img_name, _) in imageloader.dataset.dataset.samples[
                                (i * batch_size) : (i * batch_size + batch_size)
                            ]
                        ]
                    else:  # for the last batch, len of img_names not necessarily = batch_size
                        img_names = [
                            img_name.split("/")[-1]
                            for (img_name, _) in imageloader.dataset.dataset.samples[
                                (i * batch_size) : (i * batch_size + len(imgs))
                            ]
                        ]
                else:
                    if i != n_batch - 1:
                        img_names = [
                            img_name.split("/")[-1]
                            for (img_name, _) in imageloader.dataset.samples[
                                (i * batch_size) : (i * batch_size + batch_size)
                            ]
                        ]
                    else:  # for the last batch, len of img_names not necessarily = batch_size
                        img_names = [
                            img_name.split("/")[-1]
                            for (img_name, _) in imageloader.dataset.samples[
                                (i * batch_size) : (i * batch_size + len(imgs))
                            ]
                        ]

                # save output
                output_name = f"{i}.{img_names[0]}.{img_names[-1]}.npz"
                np.savez(
                    model_res_dir + "/" + output_name,
                    batch_boxes=batch_boxes,
                    batch_probs=batch_probs,
                    batch_points=batch_points,
                    age=age,
                    gender=gender,
                    emotion=emotion,
                    emotion_prob=emotion_prob,
                    img_names=img_names,
                    dtype=object,
                )
                # delete error entry if no error occurred for this batch
                if all(v == "" for v in err[i].values()):
                    del err[i]
                continue

            n_faces = []  # number of faces for each img, e.g., [0, 0, 2, 1, 0, ...]
            faces_input = (
                []
            )  # list of face tensors, e.g., [(1, 224, 224, 3), (2, 224, 224, 3), ...]
            for elem in faces:
                if isinstance(elem, torch.Tensor):
                    n_faces.append(elem.shape[0])
                    faces_input.append(elem)
                else:
                    n_faces.append(0)

            faces_input = torch.cat(faces_input, dim=0)
            with torch.no_grad():
                ag_res = age_gender_detector.detect(faces_input)
                e_res = emotion_detector.detect_emotion(faces_input)

            age = []
            gender = []
            emotion = []
            emotion_prob = []
            faces_added = 0
            for _, n in enumerate(n_faces):
                if n == 0:
                    age.append(None)
                    gender.append(None)
                    emotion.append(None)
                    emotion_prob.append(None)
                else:
                    age.append(ag_res[1][faces_added : faces_added + n])
                    gender.append(ag_res[0][faces_added : faces_added + n])
                    emotion.append(list(e_res[0][faces_added : faces_added + n]))
                    emotion_prob.append(e_res[1][faces_added : faces_added + n])
                    faces_added += n
        except Exception as e:
            t_analysis = time.time()
            analysis_log = f"\t\t Failed analyzing faces for batch {i + 1}, time elapsed: {int((t_analysis-t0)/60)}m {round((t_analysis-t0)%60, 2)}s. \n"
            print(analysis_log)
            err[i]["analysis"] = str(e)
            with open(model_log_path, "a", encoding="utf-8") as ap_log:
                ap_log.write(analysis_log)

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
                            (i * batch_size) : (i * batch_size + batch_size)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.dataset.samples[
                            (i * batch_size) : (i * batch_size + len(imgs))
                        ]
                    ]
            else:
                if i != n_batch - 1:
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.samples[
                            (i * batch_size) : (i * batch_size + batch_size)
                        ]
                    ]
                else:  # for the last batch, len of img_names not necessarily = batch_size
                    img_names = [
                        img_name.split("/")[-1]
                        for (img_name, _) in imageloader.dataset.samples[
                            (i * batch_size) : (i * batch_size + len(imgs))
                        ]
                    ]

            # save output
            output_name = f"{i}.{img_names[0]}.{img_names[-1]}.npz"
            np.savez(
                model_res_dir + "/" + output_name,
                batch_boxes=batch_boxes,
                batch_probs=batch_probs,
                batch_points=batch_points,
                age=age,
                gender=gender,
                emotion=emotion,
                emotion_prob=emotion_prob,
                img_names=img_names,
                dtype=object,
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
