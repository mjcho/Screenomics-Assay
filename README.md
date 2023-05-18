# ScrAssay

## Install required packages
Install required packages using `conda env create --file [req].txt`. This recreates the environment for the pipeline. 

Additionally, clone the following repos:
https://github.com/onnx/models.git
https://github.com/clovaai/deep-text-recognition-benchmark.git
https://github.com/clovaai/CRAFT-pytorch.git

Note that some of the packages are not available at `conda`. Download them using pip after the conda requirements are fulfilled:

`pillow-simd`: High-performance version of pillow. To install and make it utilize `libjpeg-turbo`, which is a high-performance jpeg compressor, first uninstall related image packages then reinstall `libjpeg-turbo` and compile `pillow-simd`. In short, do:
```
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
pip uninstall -y pillow
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
```
Test if `pillow-simd` is utilizing `libjpeg-turbo`:
`python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"`  
Reference:https://fastai1.fast.ai/performance.html#pillow-simd

Also, see known issues bellow to fix incompatibility with `torchvision`.  

`nvidia-tensorrt`: This is required only if using the tensorrt execution provider of `onnxruntime`. Do:
```
pip install --upgrade setuptools pip # update pip
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
```

`torch-tensorrt`: This is a torch interface to tensorrt. Importing this via jupyter causes error not finding the file libnvinfer_plugin.so.8. Solution is to copy it from the env to /usr/lib:
```
sudo cp /opt/conda/envs/cmods/lib/python3.7/site-packages/tensorrt/libnvinfer_plugin.so.8 /usr/lib
```


`deepface`  



## Download models
### ONNX
`cd models
git lfs install`

Then:  
```
git lfs pull --include="vision/body_analysis/ultraface/models/version-RFB-640.onnx" --exclude=""
git lfs pull --include="vision/body_analysis/age_gender/models/age_googlenet.onnx" --exclude=""
git lfs pull --include="vision/body_analysis/age_gender/models/gender_googlenet.onnx" --exclude=""
git lfs pull --include="vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" --exclude=""
git lfs pull --include="vision/object_detection_segmentation/yolov4/model/yolov4.onnx" --exclude=""
git lfs pull --include="vision/classification/resnet/model/resnet152-v2-7.onnx" --exclude=""
```
<!-- git lfs pull --include="vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx" --exclude="" -->

or download all models using:  
`git lfs pull --include="*" --exclude=""`  

### CRAFT-pytorch
`gdown --id 1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9`  
`gdown --id 1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY` (may not need this, depending on use)

### deep-text-recognition-benchmark
`gdown --id 1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ`


## Known issues
### torch
Possible conflict between onnx and torch versions.
In `/opt/conda/envs/[CONDA_ENV]/lib/python3.7/site-packages/torch/onnx/utils.py`, change 
`from torch._six import container_abcs` to  
`import collections.abc as container_abcs`

### torchvision
Possible conflict between torchvision and pillow versions.
In `/opt/conda/envs/[CONDA_ENV]/lib/python3.7/site-packages/torchvision/transforms/functional.py` change the fifth line to:  
```
try:
    from PIL import Image, ImageOps, ImageEnhance,PILLOW_VERSION
except:
    from PIL import Image, ImageOps, ImageEnhance
    PILLOW_VERSION="7.0.0"
```
Reference: https://support.huaweicloud.com/ug-pt-training-pytorch/atlasmprtg_13_0024.html

### pillow
To load truncated images as normal (completed with grey bits), modify line 42 of `/opt/conda/envs/tv0.11/lib/python3.7/site-packages/PIL/ImageFile.py` to:
`LOAD_TRUNCATED_IMAGES = True`
or 
```
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

<<<<<<< HEAD
### EasyOCR
To select device other than cuda:0, need to change `torch.nn.DataParallel` lines in the `easyocr` package.  
Specifically: 
```
detection.py:87: net = torch.nn.DataParallel(net, device_ids=[torch.device(device)]).to(device)
recognition.py:182: model = torch.nn.DataParallel(model, device_ids=[torch.device(device)]).to(device)
```
Ref.:https://github.com/JaidedAI/EasyOCR/issues/295

=======
>>>>>>> c4a06665f6e296ecd4b823e23e83f3da8bb9ef4d

## Notes
### GCP Deep Learning VMs
This tool was developed on GCP VMs. If you create a GPU VM (also called Deep Learning VM) and a boot disk for it from an image pre-installed with Nvidia drivers and CUDA tools, then Nvidia drivers may not be correctly identified by `nvidia-smi`. To fix this, try execute `sudo /opt/deeplearning/install-driver.sh` first then restart the VM.

