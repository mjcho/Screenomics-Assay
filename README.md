# Screenomics Assay

## Install required packages
Install required packages using `conda env create --file [req].txt`. This recreates the environment for the pipeline. 

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

## Download models
### Image classification: ONNX
Clone this: https://github.com/onnx/models.git
Then:
```cd models git lfs install
git lfs pull --include="vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx" --exclude=""
```

or download all models using:  
`git lfs pull --include="*" --exclude=""`  


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

### GCP Deep Learning VMs
This tool was developed on GCP VMs. If you create a GPU VM (also called Deep Learning VM) and a boot disk for it from an image pre-installed with Nvidia drivers and CUDA tools, then Nvidia drivers may not be correctly identified by `nvidia-smi`. To fix this, try execute `sudo /opt/deeplearning/install-driver.sh` first then restart the VM.

