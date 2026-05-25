"""Prepare training and test datasets for SRCNN."""

from pathlib import Path
import os
import cv2
import h5py
import numpy as np
import random
import shutil
try:
    from .config import *
except Exception:
    from config import *

# Load Cube RGB Normalized 
def load_cube(path):
    path = Path(path)

    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")

    return img.astype(np.float32)

# Prepare Data for Testing and Validation
def prepare_data(_path):
    folder = Path(_path)
    names = sorted([name for name in folder.iterdir() if name.is_file()])
    nums = len(names)

    data = np.zeros((nums * Random_Crop, 3, Patch_size, Patch_size), dtype=np.float32)
    label = np.zeros((nums * Random_Crop, 3, label_size, label_size), dtype=np.float32)

    for i in range(nums):
        name = names[i]
        #hr_img = cv2.imread(str(name), cv2.IMREAD_UNCHANGED)
        hr_img = load_cube(name)
        shape = hr_img.shape

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce Random_Crop random coordinate to crop training img
        Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            # lr_patch = lr_patch.astype(float) / 4095.
            # hr_patch = hr_patch.astype(float) / 4095.
            lr_patch = lr_patch.astype(np.float32)
            hr_patch = hr_patch.astype(np.float32)      

            data[i * Random_Crop + j] = np.transpose(lr_patch, (2, 0, 1))
            label[i * Random_Crop + j] = np.transpose(hr_patch[conv_side: -conv_side, conv_side: -conv_side], (2, 0, 1))
            

    return data, label

# BORDER_CUT = 8
BLOCK_STEP = 16
BLOCK_SIZE = 32

# Prepare Data for Training
def prepare_crop_data(_path):
    folder = Path(_path)
    names = sorted([name for name in folder.iterdir() if name.is_file()])
    nums = len(names)

    data = []
    label = []

    for i in range(nums):
        name = names[i]
        #hr_img = cv2.imread(str(name), cv2.IMREAD_UNCHANGED)
        hr_img = load_cube(name)
        shape = hr_img.shape

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
        height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
        for k in range(width_num):
            for j in range(height_num):
                x = k * BLOCK_STEP
                y = j * BLOCK_STEP
                hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                # lr_patch = lr_patch.astype(float) / 4095.
                # hr_patch = hr_patch.astype(float) / 4095.
                lr_patch = lr_patch.astype(np.float32)
                hr_patch = hr_patch.astype(np.float32)

                lr = np.zeros((3, Patch_size, Patch_size), dtype=np.float32)
                hr = np.zeros((3, label_size, label_size), dtype=np.float32)

                lr = np.transpose(lr_patch, (2, 0, 1))
                hr = np.transpose(hr_patch[conv_side: -conv_side, conv_side: -conv_side], (2, 0, 1))

                data.append(lr)
                label.append(hr)

    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=np.float32)
    return data, label


def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


def main():
    TRAIN_RGB.mkdir(parents=True, exist_ok=True)
    TEST_RGB.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    #all_images = sorted([f for f in RGB_NORM.iterdir() if f.suffix.lower() in VALID_EXT and f.is_file()])
    
    all_images = [
        f for f in RGB_NORM.iterdir()
        if f.suffix.lower() == ".npy" and f.is_file()
    ]

    # Only select images with non-zero values
    valid_images = []

    for f in all_images:
        arr = np.load(f)
        if arr.max() > 0:
            valid_images.append(f)

    # Select 47 + 58 images
    # selected = random.sample(all_images, 105)
    # train_imgs = selected[:91]
    # test_imgs = selected[91:]
    scene_ids = []
    for f in valid_images:

        stem = f.stem

        # RGB_<scene_id>_<patch>
        scene_id = "_".join(stem.split("_")[1:-1])

        if scene_id not in scene_ids:
            scene_ids.append(scene_id)

    random.seed(42)
    random.shuffle(scene_ids)
    split_idx = int(len(scene_ids) * 0.7)   #70% training, 30% test

    train_scene_ids = scene_ids[:split_idx]
    test_scene_ids = scene_ids[split_idx:]    

    train_imgs = []
    test_imgs = []

    for f in valid_images:

        scene_id = "_".join(f.stem.split("_")[1:-1])

        if scene_id in train_scene_ids:
            train_imgs.append(f)
        else:
            test_imgs.append(f)

    # Copy file into folder
    for img in train_imgs:
        shutil.copy(str(img), str(TRAIN_RGB / img.name))
    for img in test_imgs:
        shutil.copy(str(img), str(TEST_RGB / img.name))


    print(len(list(TRAIN_RGB.iterdir())))
    print(len(list(TEST_RGB.iterdir())))

    
    data, label = prepare_crop_data(TRAIN_RGB)
    write_hdf5(data, label, str(PROCESSED_DIR / "crop_train.h5"))
    data, label = prepare_data(TEST_RGB)
    write_hdf5(data, label, str(PROCESSED_DIR / "test.h5"))
    # _, _a = read_training_data("train.h5")
    # _, _a = read_training_data("test.h5")


if __name__ == "__main__":
    main()