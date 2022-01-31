from pathlib import Path
import argparse
import glob
import shutil
import random


"""
Make a local dataset from source images

"""

def _get_file_count(images_path):
    print(f"{images_path}/*)")
    return len(glob.glob(f"{images_path}/*"))

def _create_dataset(train_size, val_size, test_size, deer_path, background_path):
    # 1: Create local dataset directory
    # 2: Create local train/validation/test directories
    # 3: Create deer, background directories in each of the train/val/test directories
    Path.mkdir(Path("./datasets/train/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path("./datasets/train/background"), parents=True, exist_ok=True)
    Path.mkdir(Path("./datasets/validation/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path("./datasets/validation/background"), parents=True, exist_ok=True)
    Path.mkdir(Path("./datasets/test/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path("./datasets/test/background"), parents=True, exist_ok=True)


    # get list of all deer files
    deer_files = [f for f in Path(deer_path).glob('*')]
    random.shuffle(deer_files)
    train_deer_files = deer_files[:train_size]
    val_deer_files = deer_files[train_size:train_size+val_size]
    test_deer_files = deer_files[train_size+val_size:train_size+val_size+test_size]

    # copy the deer files
    for deer_file in train_deer_files:
        shutil.copy(deer_file, f"./datasets/train/deer/{deer_file.parts[-1]}")
    for deer_file in val_deer_files:
        shutil.copy(deer_file, f"./datasets/validation/deer/{deer_file.parts[-1]}")
    for deer_file in test_deer_files:
        shutil.copy(deer_file, f"./datasets/test/deer/{deer_file.parts[-1]}")

    # get list of all background files
    background_files = [f for f in Path(background_path).glob('*')]
    random.shuffle(background_files)
    train_bkgd_files = background_files[:train_size]
    val_bkgd_files = background_files[train_size:train_size+val_size]
    test_bkgd_files = background_files[train_size+val_size:train_size+val_size+test_size]

    # copy the deer files
    for bkgd_file in train_bkgd_files:
        shutil.copy(bkgd_file, f"./datasets/train/background/{bkgd_file.parts[-1]}")
    for bkgd_file in val_bkgd_files:
        shutil.copy(bkgd_file, f"./datasets/validation/background/{bkgd_file.parts[-1]}")
    for bkgd_file in test_bkgd_files:
        shutil.copy(bkgd_file, f"./datasets/test/background/{bkgd_file.parts[-1]}")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deer-images", type=str, required=False, default="../datasets/deer", help="path to deer images")
    ap.add_argument("--background-images", type=str, required=False, default="../datasets/landscape",
                    help="path to background images")
    ap.add_argument("--train-val-test", type=str, required=False, default="70,20,10",
                    help="Percentage split between train, validation and test ")

    args = vars(ap.parse_args())

    deer_path = args['deer_images']
    background_path = args['background_images']
    tvt_split = args['train_val_test']
    tvt_split = tvt_split.split(",") # 0-train, 1-val, 2-test

    num_deer_files = _get_file_count(deer_path)
    num_landscape_files = _get_file_count(background_path)
    train_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[0])/100)
    val_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[1])/100)
    test_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[2])/100)

    print(train_size, val_size, test_size)
    _create_dataset(train_size, val_size, test_size, deer_path, background_path)




if __name__ == '__main__':
    _main()
