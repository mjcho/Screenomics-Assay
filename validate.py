import argparse
import os
import sys
import imghdr
import time
from datetime import datetime
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def img_verify2_p(path):
    if not imghdr.what(path):
        return path
    else:
        return None


# multithreading tqdm func
def run(f, filepaths):
    with ThreadPoolExecutor(max_workers=20) as executor:
        is_invalid = list(tqdm(executor.map(f, filepaths), total=len(filepaths)))
    return is_invalid


def validate(args):
    parser = argparse.ArgumentParser(description="Validate the images in parallel.")
    parser.add_argument(
        "dirpath",
        help="the directory of the images (relative to user dir), first level needs to be classes/participants",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="the directory for output log and invalid images (relative to user dir), default is user dir",
        default=os.path.expanduser("~"),
    )

    args = parser.parse_args()
    # dirpath = os.path.expanduser(f"~/{args.dirpath}")
    # out_dir = os.path.expanduser(f"~/{args.out_dir}")
    dirpath = args.dirpath
    out_dir = args.out_dir

    # list files in the directory
    log_s = f"Listing images for {dirpath} at: {str(datetime.now())}.\n"
    print(log_s)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_s)

    filepaths = []
    t_list = time.time()
    for root, dirs, files in os.walk(dirpath):
        for d in dirs:
            files_in_dir = os.listdir(os.path.join(root, d))
            # print(f'\tListing for {d} done in : {time.time()-t_list}s.')
            filepaths_in_dir = [os.path.join(root, d, f) for f in files_in_dir]
            # print(f'\tJoining paths for {d} done in : {time.time()-t_list}s.')
            filepaths.append(filepaths_in_dir)
            log_ld = f"\tListing for {d} done in : {time.time()-t_list}s.\n"
            print(log_ld)
            with open(
                f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8"
            ) as log:
                log.write(log_ld)

    log_le = f"Listing ends for {dirpath} at: {str(datetime.now())}.\n"
    print(log_le)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_le)

    # concat dir-level lists to one
    log_cs = (
        f"Concatenating file lists for {dirpath} starts at: {str(datetime.now())}.\n"
    )
    print(log_cs)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_cs)

    filepaths = [p for l in filepaths for p in l]  # concat lists of files into one

    log_ce = f"Concatenating file lists for {dirpath} ends at: {str(datetime.now())}. List len = {len(filepaths)}.\n"
    print(log_ce)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_ce)

    # multithreading image verification
    log_vs = f"Verification for {dirpath} starts at: {str(datetime.now())}.\n"
    print(log_vs)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_vs)

    is_invalid = run(img_verify2_p, filepaths)

    log_ve = f"Verification for {dirpath} ends at: {str(datetime.now())}.\n"
    print(log_ve)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_ve)

    # save list of invalid images
    log_ss = f"Saving for {dirpath} starts at: {str(datetime.now())}.\n"
    print(log_ss)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_ss)

    invalid_images = [i for i in is_invalid if i]  # a list only of the invalid images
    with open(f"{out_dir}/invalid_images_{dirpath.split('/')[-1]}.pkl", "wb") as f:
        pickle.dump(invalid_images, f, -1)

    log_se = f"Saving for {dirpath} ends at: {str(datetime.now())}.\n\n"
    print(log_se)
    with open(f"{out_dir}/invalid_checked_test.txt", "a", encoding="utf-8") as log:
        log.write(log_se)


if __name__ == "__main__":
    validate(sys.argv)
