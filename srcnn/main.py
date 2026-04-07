from pathlib import Path

import cv2
import math
import numpy
import shutil
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D, Input
from keras.models import Sequential
from keras.optimizers import Adam, SGD

try:
    from . import prepare_data as pd
except ImportError:
    import prepare_data as pd


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs" / "Result"
ASSETS_DIR = ROOT / "assets"


def model():
    SRCNN = Sequential()

    # First layer: 128 filters, 9x9 kernel, ReLU activation, valid padding
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu',
                     kernel_initializer='glorot_uniform', padding='valid', input_shape=(32, 32, 3)))
    # Second layer: 64 filters, 3x3 kernel, ReLU activation, same padding
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='glorot_uniform', padding='same'))
    # Third layer: 3 filters, 5x5 kernel, linear activation, valid padding
    SRCNN.add(Conv2D(filters=3, kernel_size=(5, 5), activation='linear',
                     kernel_initializer='glorot_uniform', padding='valid'))
    adam = Adam(learning_rate=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict_model():

    SRCNN = Sequential()

    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu',
                     kernel_initializer='glorot_uniform', padding='valid', input_shape=(None, None, 3)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     kernel_initializer='glorot_uniform', padding='same'))
    SRCNN.add(Conv2D(filters=3, kernel_size=(5, 5), activation='linear',
                     kernel_initializer='glorot_uniform', padding='valid'))

    adam = Adam(learning_rate=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train():
    srcnn_model = model()
    print(srcnn_model.summary())

    data, label = pd.read_training_data(str(PROCESSED_DATA_DIR / "crop_train.h5"))
    val_data, val_label = pd.read_training_data(str(PROCESSED_DATA_DIR / "test.h5"))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(str(WEIGHTS_DIR / "SRCNN_check.h5"), 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True,
                                 save_weights_only=False, 
                                 mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, 
                    batch_size=128, 
                    validation_data=(val_data, val_label),
                    callbacks=callbacks_list, 
                    shuffle=True, 
                    epochs=200, 
                    verbose=0
                    )
    # srcnn_model.load_weights("m_model_adam.h5")


def predict():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    srcnn_model = predict_model()
    srcnn_model.load_weights(str(WEIGHTS_DIR / "SRCNN_check.h5"))
    IMG_NAME = ASSETS_DIR / "input2.jpg"            #DOVREBBE ESSERE .TIF O .JP2 MA PER OR ANON HO DATI
    INPUT_NAME = OUTPUTS_DIR / "input2.png"
    OUTPUT_NAME = OUTPUTS_DIR / "pre2.png"

    # Original image is used as reference for PSNR calculation
    REFERENCE_NAME = OUTPUTS_DIR / "reference.png"  #Devi modificare dipende dall'immagine che vuoi paragonare
    shutil.copy(IMG_NAME, REFERENCE_NAME)
    img_reference = cv2.imread(str(IMG_NAME), cv2.IMREAD_UNCHANGED)

    if len(img_reference.shape) != 3 or img_reference.shape[2] != 3:
        raise ValueError(f"L'immagine {IMG_NAME} non è RGB a 3 canali")

    # Save the reference image for PSNR calculation
    shape = img_reference.shape

    # Degradation using bicubic interpolation
    # 1. Converto to float32 for processing
    img_float = img_reference.astype(numpy.float32)
    # 2. Downscale by a factor of 2 using bicubic interpolation
    lr_img = cv2.resize(
        img_float,
        (shape[1] // 2, shape[0] // 2),
        interpolation=cv2.INTER_CUBIC
    )
    # 3. Upscale back to original size using bicubic interpolation
    bicubic_img = cv2.resize(
        lr_img,
        (shape[1], shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    # 4. Save the bicubic image for PSNR calculation (CLIPPING + CASTING)
    bicubic_to_save = numpy.clip(bicubic_img, 0, 4095.0).astype(numpy.uint16)
    cv2.imwrite(str(INPUT_NAME), bicubic_to_save)

    # 5. Normalize the bicubic image to [0, 1] range for SRCNN input
    X = bicubic_img / 4095.0
    X = numpy.expand_dims(X, axis=0).astype(numpy.float32)   # (1, H, W, 3)

    # 6. Predict the high-resolution image using SRCNN
    pre = srcnn_model.predict(X, batch_size=1)

    # 7. Denormalize the output back to [0, 4095] range and save
    pre = pre * 4095.0
    pre = numpy.clip(pre, 0, 4095.0)
    pre = pre.astype(numpy.uint16)

    # 8. Save the predicted image
    cv2.imwrite(str(OUTPUT_NAME), pre[0])


if __name__ == "__main__":
    train()
    predict()
