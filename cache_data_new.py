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
# from mocov3 import vits

from PIL import features, Image
from packaging import version

try:    ver = Image.__version__     # PIL >= 7
except: ver = Image.PILLOW_VERSION  # PIL <  7

if version.parse(ver) >= version.parse("5.4.0"):
    if features.check_feature('libjpeg_turbo'):
        print("libjpeg-turbo is on")
    else:
        print("libjpeg-turbo is not on")
else:
    print(f"libjpeg-turbo' status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {ver}")

def get_model(model_name, moco_pth='/home/ezelikman/vit-b-300ep.pth.tar', device='cuda:0'):
    print(f"Getting model {model_name}")
    if "clip" in model_name:
        models = ['RN50x16', 'ViT-B/16', 'ViT-L/14']
        model, _ = clip.load(models[2], device=device)
        encode_image = model.encode_image
    elif "dino_" in model_name:
        model = torch.hub.load('facebookresearch/dino:main', model_name)
        model = torch.nn.DataParallel(model)
        encode_image = model
    elif model_name == "moco":
        model = vits.vit_base()
        checkpoint = torch.load(moco_pth)
        state_dict = checkpoint['state_dict']
        linear_keyword = 'head'
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.head = nn.Identity()
        encode_image = model
    else:
        model = timm.create_model(model_name, pretrained=True)
        encode_image = lambda x: model.forward_features(x).mean(-1).mean(-1)
    return model.to(device), encode_image


# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
class SquarePad:
    def __call__(self, image):
        image_shape = image[0].shape
        max_wh = max(image_shape)
        p_left, p_top = [(max_wh - s) // 2 for s in image_shape]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image_shape, [p_left, p_top])]
        padding = (p_top, p_left, p_bottom, p_right)
        return F.pad(image, padding, 0, 'constant')

transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    SquarePad(),
    transforms.Resize(size=224),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model_name', help='the name of the model to use')
    parser.add_argument("-d", "--device", help="specify device id", default="cuda:0")
    parser.add_argument("-r", "--root", help="specify image root dir in full", default="/home/family")
    parser.add_argument("-o", "--output_dir", help="specify output dir in full", default="/home/mjcho")
    parser.add_argument("-bs", "--batch_size", help="specify batch size", type=int, default = 1024)
    parser.add_argument("-w", "--workers", help="specify number of workers", type=int, default = 12)
    # parser.add_argument("-rs", "--resume_start", help="specify index of image to resume from", type=int)
    args = parser.parse_args()

    model_name = args.model_name
    device = args.device
    root = args.root
    output_dir = args.output_dir
    batch_size = args.batch_size
    workers = args.workers
    # resume_start = args.resume_start
    
    model, encode_image = get_model(model_name, device=device)    
    print(f"Finished getting model {model_name}")
    
    # load dataset
    folder_name = root.split('/')[-1] # 'family'
    image_dataset_file = f"{output_dir}/{folder_name}_dataset_FeatureExtractors.pkl"
    if not os.path.exists(image_dataset_file):
        print(f"Saving {image_dataset_file}")
        image_dataset = ImageFolder(root=root, transform=transform)
        with open(image_dataset_file, 'wb') as f:
            pickle.dump(image_dataset, f, -1)
    else:
        print(f"Loading {image_dataset_file}")
        with open(image_dataset_file, 'rb') as infile:
            image_dataset = pickle.load(infile)
    
    # save file that stores img order
    model_output_dir = f"{output_dir}/assay_output_{folder_name}/{model_name}"
    img_order_file = f"{output_dir}/assay_output_{folder_name}/feature_extractors_filenames.pkl"
    img_order = image_dataset.imgs
    os.makedirs(model_output_dir, exist_ok=True)    
    if not os.path.exists(img_order_file):
        print(f"Saving {img_order_file}")
        with open(img_order_file, 'wb') as fp:
            pickle.dump(img_order, fp)

    # save every save_batch images
    save_batch = 65536
    
    # # subset if resuming
    # if resume_start:
    #     image_dataset = torch.utils.data.Subset(image_dataset, range(new_start, len(image_dataset)))
    #     file_counter = int(resume_start / save_batch)
    #     print(f"Resuming from {resume_start}, file no. {file_counter}")
    # else:
    #     file_counter = 0
    
    # resume from where is left off by checking the saved files
    cur_files = os.listdir(model_output_dir)
    max_idx = max([int(cur_file.split(".")[0].split("_")[-1]) for cur_file in cur_files] + [0])
    if max_idx != 0:
        image_dataset = torch.utils.data.Subset(image_dataset, range(max_idx, len(image_dataset)))
        print(f"Starting from {max_idx} of {len(image_dataset)}")
    
    # create dataloader
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, num_workers=workers)

    def save_batch_array(accumulated_data):
        return np.savez_compressed(
            f"{model_output_dir}/{file_counter * save_batch}_{file_counter * save_batch + len(accumulated_data)}.npz",
            embeddings=accumulated_data
        )
    
    file_counter = max_idx // save_batch
    accumulated_data = None            
    with torch.no_grad():
        for image_idx, (image, _) in enumerate(tqdm(image_dataloader)):
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
        if accumulated_data is not None: # for saving last batches
            accumulated_data = torch.cat(accumulated_data).contiguous()
            save_batch_array(accumulated_data)
            accumulated_data = None
            file_counter += 1
    
