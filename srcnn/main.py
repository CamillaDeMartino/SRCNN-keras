from pathlib import Path

import cv2
import h5py
import math
import numpy
import shutil
import random

from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D, Input, ReLU
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
try:
    from .config import *
except Exception:
    from config import *

try:
    from .psnr import compute_psnr
except Exception:
    from psnr import compute_psnr


class HDF5Sequence(Sequence):
    def __init__(self, file_path, batch_size=128, shuffle=False, max_samples=None):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._file = None

        with h5py.File(self.file_path, "r") as h5_file:
            self.length = h5_file["data"].shape[0]

        if max_samples is not None:
            self.length = min(self.length, max_samples)

        self.indices = numpy.arange(self.length)
        self.on_epoch_end()

        # with h5py.File(self.file_path, "r") as h5_file:
        #     real_length = h5_file["data"].shape[0]

        # if max_samples is not None:
        #     max_samples = min(real_length, max_samples)
        #     self.indices = numpy.random.default_rng(42).choice(
        #         real_length,
        #         size=max_samples,
        #         replace=False
        #     )
        #     self.length = max_samples
        # else:
        #     self.length = real_length
        #     self.indices = numpy.arange(real_length)

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __getitem__(self, index):
        self._open()

        batch_ids = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = numpy.sort(batch_ids)
        x = self._file["data"][batch_ids]
        y = self._file["label"][batch_ids]

        x = numpy.transpose(x, (0, 2, 3, 1)).astype(numpy.float32)
        y = numpy.transpose(y, (0, 2, 3, 1)).astype(numpy.float32)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            numpy.random.shuffle(self.indices)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


def model():
    SRCNN = Sequential()

    # First layer: 128 filters, 9x9 kernel, ReLU activation, valid padding
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation=None,
                     kernel_initializer='glorot_uniform', padding='valid', input_shape=(Patch_size, Patch_size, 3)))
    SRCNN.add(ReLU())
    # Second layer: 64 filters, 3x3 kernel, ReLU activation, same padding
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation=None,
                     kernel_initializer='glorot_uniform', padding='same'))
    SRCNN.add(ReLU())
    # Third layer: 3 filters, 5x5 kernel, linear activation, valid padding
    SRCNN.add(Conv2D(filters=3, kernel_size=(5, 5), activation='linear',
                     kernel_initializer='glorot_uniform', padding='valid'))
    adam = Adam(learning_rate=1e-4)
    SRCNN.compile(
        optimizer=adam,
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
        jit_compile=False,
    )
    return SRCNN


def predict_model():

    SRCNN = Sequential()

    # This is the difference from model() to allow variable input size
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation=None,
                     kernel_initializer='glorot_uniform', padding='valid', input_shape=(None, None, 3)))   
    SRCNN.add(ReLU())

    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation=None,
                     kernel_initializer='glorot_uniform', padding='same'))
    SRCNN.add(ReLU())
    SRCNN.add(Conv2D(filters=3, kernel_size=(5, 5), activation='linear',
                     kernel_initializer='glorot_uniform', padding='valid'))

    adam = Adam(learning_rate=1e-4)
    SRCNN.compile(
        optimizer=adam,
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
        jit_compile=False,
    )
    return SRCNN


def train():
    srcnn_model = model()
    print(srcnn_model.summary())


    train_sequence = HDF5Sequence(PROCESSED_DIR / "train.h5", batch_size=8, shuffle=True, max_samples=2000000)
    val_sequence = HDF5Sequence(PROCESSED_DIR / "validation.h5", batch_size=8, shuffle=False, max_samples=200000)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(str(WEIGHTS_DIR / "train_global_norm_2M.h5"), 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True,
                                 save_weights_only=False, 
                                 mode='min')

    # Early stopping + LR reduction to avoid wasted epochs when validation stops improving
    earlystop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    callbacks = [checkpoint, earlystop, reduce_lr]

    try:
        srcnn_model.fit(train_sequence, 
                        validation_data=val_sequence,
                        callbacks=callbacks, 
                        shuffle=False, 
                        epochs=15, 
                        verbose=0
                        )
    finally:
        # ensure file handles are closed
        train_sequence.close()
        val_sequence.close()


def predict():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the trained model
    srcnn_model = predict_model()
    srcnn_model.load_weights(str(WEIGHTS_DIR / "train_global_norm_2M.h5"))
    print("Model loaded successfully.")

    # Original image HR (from test set) is used as reference for PSNR calculation
    # test_files = sorted(TEST_RGB.glob("*.npy"))
    # if not test_files:
    #     raise RuntimeError(f"Nessun file .npy trovato in {TEST_RGB}")
    # #random.seed(42)
    # input_path = random.choice(test_files)
    input_path = TEST_RGB / "RGB_LC08_L1TP_193029_20260424_20260424_02_RT_0069.npy"

    hr_img = numpy.load(input_path).astype(numpy.float32)
    hr_img = numpy.clip(hr_img, 0.0, 1.0)

    if len(hr_img.shape) != 3 or hr_img.shape[2] != 3:
        raise ValueError(f"L'immagine {input_path} non è RGB a 3 canali")

    # Save the reference image for PSNR calculation
    shape = hr_img.shape

    # HR image degradation using bicubic interpolation
    # 1. Downscale by a factor of 2 using bicubic interpolation
    lr_img = cv2.resize(
        hr_img,
        (shape[1] // scale, shape[0] // scale),
        interpolation=cv2.INTER_CUBIC
    )

    # 2. Upscale back to original size using bicubic interpolation
    bicubic_img = cv2.resize(
        lr_img,
        (shape[1], shape[0]),
        interpolation=cv2.INTER_CUBIC
    ).astype(numpy.float32)

    # 3. Save the bicubic image for PSNR calculation (CLIPPING + CASTING)
    bicubic_img = numpy.clip(bicubic_img, 0.0, 1.0)
    bicubic_crop = bicubic_img[conv_side:-conv_side, conv_side:-conv_side, :]
    X = numpy.expand_dims(bicubic_img, axis=0).astype(numpy.float32)   # (1, H, W, 3)

    # 4. Predict the high-resolution image using SRCNN
    residual_pred = srcnn_model.predict(X, batch_size=1)[0]
    pre = bicubic_crop + residual_pred
    pre_clip = numpy.clip(pre, 0.0, 1.0)

    # 5. Output SRCNN è più piccolo per via del valid padding: 512 -> 500 circa
    ref = hr_img[conv_side:-conv_side, conv_side:-conv_side, :]

    # 6. Save results
    numpy.save(ASSETS_DIR / "reference_b.npy", ref)
    numpy.save(ASSETS_DIR / "bicubic_b.npy", bicubic_crop)
    numpy.save(ASSETS_DIR / "srcnn_pred_b.npy", pre)
    numpy.save(ASSETS_DIR / "srcnn_pred_clip_b.npy", pre_clip)

    print("Input:", input_path)
    print("HR reference:", ref.shape, ref.dtype, ref.min(), ref.max())
    print("Bicubic:", bicubic_crop.shape, bicubic_crop.dtype, bicubic_crop.min(), bicubic_crop.max())
    print("Prediction:", pre.shape, pre.dtype, pre.min(), pre.max())
    print("Prediction mean:", pre.mean())
    print("Reference mean:", ref.mean())    



def evaluate_many(num_images=100):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    srcnn_model = predict_model()
    srcnn_model.load_weights(str(WEIGHTS_DIR / "train_global_norm_2M.h5"))
    print("Model loaded successfully.")

    test_files = sorted(TEST_RGB.glob("*.npy"))

    if not test_files:
        raise RuntimeError(f"Nessun file .npy trovato in {TEST_RGB}")

    random.seed(42)
    selected_files = random.sample(test_files, min(num_images, len(test_files)))

    psnr_bicubic_values = []
    psnr_srcnn_values = []
    results = []
    textures = []

    for idx, input_path in enumerate(selected_files, start=1):
        hr_img = numpy.load(input_path).astype(numpy.float32)

        if hr_img.ndim != 3 or hr_img.shape[2] != 3:
            print(f"Skip invalid shape: {input_path}")
            continue

        shape = hr_img.shape

        ref = numpy.clip(
            hr_img[conv_side:-conv_side, conv_side:-conv_side, :],
            0.0,
            1.0
        )


        # Skip images where the reference is mostly empty (e.g. all black)
        if ref.mean() < 0.001 or numpy.mean(ref > 0.001) < 0.01:
            print(f"Skip mostly empty image: {input_path.name}")
            continue

        valid_mask =numpy.mean(ref, axis=2) > 0.001
        valid_fraction = numpy.mean(valid_mask)

        if valid_fraction < 0.98:
            print(
                f"Skip image with too much no-data: "
                f"{input_path.name} | valid_fraction={valid_fraction:.4f}"
            )
            continue

        texture = ref.std()
        if texture < 0.02:
            print(f"Skip image with low texture: {input_path.name}")
            continue
        textures.append(texture)

        lr_img = cv2.resize(
            hr_img,
            (shape[1] // scale, shape[0] // scale),
            interpolation=cv2.INTER_CUBIC
        )

        bicubic_img = cv2.resize(
            lr_img,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_CUBIC
        ).astype(numpy.float32)

        bicubic_img = numpy.clip(bicubic_img, 0.0, 1.0)
        bicubic_crop = bicubic_img[conv_side:-conv_side, conv_side:-conv_side, :]

        X = numpy.expand_dims(bicubic_img, axis=0).astype(numpy.float32)

        residual_pred = srcnn_model.predict(X, batch_size=1, verbose=0)[0]

        pred = bicubic_crop + residual_pred
        pred = numpy.clip(pred, 0.0, 1.0)

        psnr_bicubic = compute_psnr(bicubic_crop, ref, max_val=1.0)
        psnr_srcnn = compute_psnr(pred, ref, max_val=1.0)

        psnr_bicubic_values.append(psnr_bicubic)
        psnr_srcnn_values.append(psnr_srcnn)

        print(
            f"[{idx}/{len(selected_files)}] "
            f"Bicubic={psnr_bicubic:.4f} dB | "
            f"SRCNN={psnr_srcnn:.4f} dB | "
            f"Gain={psnr_srcnn - psnr_bicubic:.4f} dB | "
            f"File: {input_path.name}"
        )

        results.append({
            "file": input_path.name,
            "psnr_bicubic": psnr_bicubic,
            "psnr_srcnn": psnr_srcnn,
            "gain": psnr_srcnn - psnr_bicubic,
            "texture": texture,
            "valid_fraction": valid_fraction,
        })


    psnr_bicubic_values = numpy.array(psnr_bicubic_values, dtype=numpy.float32)
    psnr_srcnn_values = numpy.array(psnr_srcnn_values, dtype=numpy.float32)
    gains = psnr_srcnn_values - psnr_bicubic_values

    print("\n=== Evaluation summary ===")
    print(f"Images evaluated: {len(psnr_bicubic_values)}")
    print(f"Mean PSNR Bicubic: {psnr_bicubic_values.mean():.4f} dB")
    print(f"Mean PSNR SRCNN:   {psnr_srcnn_values.mean():.4f} dB")
    print(f"Mean Gain:         {gains.mean():.4f} dB")
    print(f"Median Gain:       {numpy.median(gains):.4f} dB")
    print(f"SRCNN better on:   {numpy.mean(gains > 0) * 100:.2f}% images")


    results = sorted(results, key=lambda x: x["gain"])
    print("\n=== WORST 10 ===")
    for r in results[:10]:
        print(r)

    print("\n=== BEST 10 ===")
    for r in results[-10:]:
        print(r)

    save_top_results(
        results=results,
        source_dir=TEST_RGB,
        output_dir=ASSETS_DIR / "best_10_cases",
        srcnn_model=srcnn_model,
        top_k=10,
        prefix="best"
    )


def save_top_results(results, source_dir, output_dir, srcnn_model, top_k=10, prefix="best"):
    """
    Per ogni top-k risultato salva:
    - reference.npy
    - bicubic.npy
    - srcnn_pred.npy
    - srcnn_pred_clip.npy
    - original.npy
    - summary.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    top_results = sorted(results, key=lambda x: x["gain"], reverse=True)[:top_k]
    summary_path = output_dir / "summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"TOP {top_k} RESULTS ({prefix.upper()})\n\n")

        for rank, r in enumerate(top_results, start=1):
            input_path = source_dir / r["file"]

            case_dir = output_dir / (
                f"{prefix}_{rank:02d}_"
                f"gain_{r['gain']:.4f}_"
                f"{Path(r['file']).stem}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)

            hr_img = numpy.load(input_path).astype(numpy.float32)
            hr_img = numpy.clip(hr_img, 0.0, 1.0)

            shape = hr_img.shape

            lr_img = cv2.resize(
                hr_img,
                (shape[1] // scale, shape[0] // scale),
                interpolation=cv2.INTER_CUBIC
            )

            bicubic_img = cv2.resize(
                lr_img,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_CUBIC
            ).astype(numpy.float32)

            bicubic_img = numpy.clip(bicubic_img, 0.0, 1.0)
            bicubic_crop = bicubic_img[conv_side:-conv_side, conv_side:-conv_side, :]

            X = numpy.expand_dims(bicubic_img, axis=0).astype(numpy.float32)

            residual_pred = srcnn_model.predict(X, batch_size=1, verbose=0)[0]

            pre = bicubic_crop + residual_pred
            pre_clip = numpy.clip(pre, 0.0, 1.0)

            ref = hr_img[conv_side:-conv_side, conv_side:-conv_side, :]

            numpy.save(case_dir / "reference.npy", ref)
            numpy.save(case_dir / "bicubic.npy", bicubic_crop)
            numpy.save(case_dir / "srcnn_pred.npy", pre)
            numpy.save(case_dir / "srcnn_pred_clip.npy", pre_clip)

            f.write(
                f"Rank: {rank}\n"
                f"Original file: {r['file']}\n"
                f"Case folder: {case_dir.name}\n"
                f"PSNR Bicubic: {r['psnr_bicubic']:.4f}\n"
                f"PSNR SRCNN:   {r['psnr_srcnn']:.4f}\n"
                f"Gain:         {r['gain']:.4f}\n"
            )

            if "texture" in r:
                f.write(f"Texture:      {r['texture']:.6f}\n")

            if "valid_fraction" in r:
                f.write(f"Valid frac:   {r['valid_fraction']:.6f}\n")

            f.write("\n")

    print(f"Saved top {top_k} prediction cases in: {output_dir}")
    print(f"Summary saved in: {summary_path}")


if __name__ == "__main__":
    #train()
    #predict()
    evaluate_many(num_images=500)