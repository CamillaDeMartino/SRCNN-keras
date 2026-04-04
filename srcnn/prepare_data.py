"""Prepare training and test datasets for SRCNN."""

from pathlib import Path
import os
import cv2
import h5py
import numpy
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train" / "Train_RGB"
TEST_PATH = ROOT / "data" / "test" / "Test_RGB"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_RGB_FALLBACK = Path.home() / "Scrivania" / "Tesi" / "RGB"


Random_Crop = 30
Patch_size = 32
label_size = 20
conv_side = 6
scale = 2


def prepare_data(_path):
    folder = Path(_path)
    names = sorted([name for name in folder.iterdir() if name.is_file()])
    nums = len(names)

    data = numpy.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=numpy.double)
    label = numpy.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=numpy.double)

    for i in range(nums):
        name = names[i]
        hr_img = cv2.imread(str(name), cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce Random_Crop random coordinate to crop training img
        Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * Random_Crop + j, 0, :, :] = lr_patch
            label[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label

# BORDER_CUT = 8
BLOCK_STEP = 16
BLOCK_SIZE = 32


def prepare_crop_data(_path):
    folder = Path(_path)
    names = sorted([name for name in folder.iterdir() if name.is_file()])
    nums = len(names)

    data = []
    label = []

    for i in range(nums):
        name = names[i]
        hr_img = cv2.imread(str(name), cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
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

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = numpy.zeros((1, Patch_size, Patch_size), dtype=numpy.double)
                hr = numpy.zeros((1, label_size, label_size), dtype=numpy.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                data.append(lr)
                label.append(hr)

    data = numpy.array(data, dtype=float)
    label = numpy.array(label, dtype=float)
    return data, label


def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


if __name__ == "__main__":
    RGB_FOLDER = RAW_RGB_FALLBACK if RAW_RGB_FALLBACK.exists() else ROOT / "data" / "raw" / "RGB"
    TRAIN_RGB = DATA_PATH
    TEST_RGB = TEST_PATH

    TRAIN_RGB.mkdir(parents=True, exist_ok=True)
    TEST_RGB.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in RGB_FOLDER.iterdir() if f.suffix.lower() == ".png"])
    
    # Select 47 + 58 images
    selected = random.sample(all_images, 105)
    train_imgs = selected[:91]
    test_imgs = selected[91:]
    

    # Copy file into folder
    for img in train_imgs:
        shutil.copy(str(img), str(TRAIN_RGB / img.name))
    for img in test_imgs:
        shutil.copy(str(img), str(TEST_RGB / img.name))


    print(len(list(TRAIN_RGB.iterdir())))
    print(len(list(TEST_RGB.iterdir())))

    
    data, label = prepare_crop_data(DATA_PATH)
    write_hdf5(data, label, str(PROCESSED_DATA_PATH / "crop_train.h5"))
    data, label = prepare_data(TEST_PATH)
    write_hdf5(data, label, str(PROCESSED_DATA_PATH / "test.h5"))
    # _, _a = read_training_data("train.h5")
    # _, _a = read_training_data("test.h5")
