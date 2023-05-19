import os
import torchvision
from torchvision import transforms
from datetime import datetime
import pickle

# Set params
module_name = "DeepAffectModule"

# set proc id and dataset
# subset_id = 0
proc_id = 0
gpu_id = 0  # int(np.floor(proc_id/2))

dataset_id = "Sarah"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# Create dataset
def create(dirpath, out_dir):
    dataset_name = dirpath.split("/")[-1]
    module = "DeepAffect"

    # create dir for saving dataset
    if not os.path.isdir(out_dir):  # for outputs
        os.mkdir(out_dir)

    # transform function
    tf = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.NEAREST
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

    globals()[dataset_name + "_dataset"] = torchvision.datasets.ImageFolder(
        dirpath, transform=tf
    )

    print(
        f"Initiation {dataset_name} {module} dataset ends at: {str(datetime.now())}\n"
    )
    with open(f"{out_dir}/dataset.log", "a", encoding="utf-8") as log:
        log.write(
            f"Initiation {dataset_name} {module} dataset ends at: {str(datetime.now())}\n"
        )

    with open(f"{out_dir}/{dataset_name}_dataset_{module}.pkl", "wb") as f:
        pickle.dump(globals()[dataset_name + "_dataset"], f, -1)


def run(dirpath, out_dir):
    print("run")
