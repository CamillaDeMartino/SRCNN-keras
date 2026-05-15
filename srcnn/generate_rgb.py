import os
import re
import glob
import cv2
import numpy as np

try:
    from .config import *
except Exception:
    from config import *


try:
    import cupy as cp
    print("CuPy importato con successo. Le operazioni saranno accelerate via GPU.")
    _use_gpu = True
except ImportError:
    print("CuPy non trovato. Le operazioni saranno eseguite su CPU (NumPy).")
    _use_gpu = False


PATCH_SIZE = 512
BANDS_REQUIRED = {"B4", "B5", "B6"}


def read_12bit_image(path):
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Impossibile leggere il file: {path}")

    if img.ndim == 3:
        img = img[:, :, 0]

    if img.dtype != np.uint16:
        img = img.astype(np.uint16)

    return img


def binning2x2_uint16(image):
    
    h, w = image.shape

    h_even = h - (h % 2)
    w_even = w - (w % 2)

    image = image[:h_even, :w_even]

    if _use_gpu:
        try:
            # Trasferisci l'immagine alla GPU
            image_gpu = cp.asarray(image)

            # Esegui il binning su GPU
            binned_gpu = (
                image_gpu[0::2, 0::2].astype(cp.uint32) +
                image_gpu[1::2, 0::2].astype(cp.uint32) +
                image_gpu[0::2, 1::2].astype(cp.uint32) +
                image_gpu[1::2, 1::2].astype(cp.uint32)
            ) // 4

            # Ritorna l'immagine alla CPU come uint16
            return cp.asnumpy(binned_gpu).astype(np.uint16)
        except Exception as e:
            print(f"Errore durante l'esecuzione su GPU: {e}. Fallback a CPU (NumPy).")
            # Se c'è un errore con CuPy, esegui su CPU
            pass # Continua all'implementazione CPU

    # Implementazione CPU (NumPy)
    binned_cpu = (
        image[0::2, 0::2].astype(np.uint32) +
        image[1::2, 0::2].astype(np.uint32) +
        image[0::2, 1::2].astype(np.uint32) +
        image[1::2, 1::2].astype(np.uint32)
    ) // 4

    return binned_cpu.astype(np.uint16)


def extract_scene_id_and_band(filename):
    """
    Supporta:
    LC08_L1TP_182043_20260427_20260427_02_RT_B4.tif
    T15WWS_20251013T181331_B04.jp2
    """
    base = os.path.basename(filename)

    # Landsat
    match_landsat = re.match(
        r"(LC0[89]_L1[TG]P_\d{6}_\d{8}_\d{8}_\d{2}_(RT|T1))_(B[45678])\.(tif|tiff)$",
        base,
        re.IGNORECASE
    )

    if match_landsat:
        scene_id = match_landsat.group(1)
        band = match_landsat.group(3).upper() # Corrected from group(2) to group(3)
        return scene_id, band

    #JP2
    match_jp2 = re.match(
        r"(.+?)_(B0?[45678])\.jp2$",
        base,
        re.IGNORECASE
    )

    if match_jp2:
        scene_id = match_jp2.group(1)
        band = match_jp2.group(2).upper().replace("B0", "B")
        return scene_id, band

    return None, None


def group_triplets(input_folder):
    """
    Raggruppa automaticamente le immagini per scena.
    Tiene solo B4, B5, B6.
    Ignora B7 e B8.
    """
    files = glob.glob(os.path.join(input_folder, "*.tif"))
    files += glob.glob(os.path.join(input_folder, "*.TIF"))
    files += glob.glob(os.path.join(input_folder, "*.tiff"))
    files += glob.glob(os.path.join(input_folder, "*.TIFF"))
    files += glob.glob(os.path.join(input_folder, "*.jp2"))
    files += glob.glob(os.path.join(input_folder, "*.JP2"))

    scenes = {}
    print(f"Trovati {len(files)} file immagine.")

    for path in files:
        scene_id, band = extract_scene_id_and_band(path)
        print(f"Processing: {os.path.basename(path)}, Extracted: Scene ID={scene_id}, Band={band}")

        if scene_id is None or band is None:
            continue

        if band not in BANDS_REQUIRED:
            continue

        scenes.setdefault(scene_id, {})
        scenes[scene_id][band] = path

    complete_triplets = {
        scene_id: bands
        for scene_id, bands in scenes.items()
        if BANDS_REQUIRED.issubset(bands.keys())
    }

    incomplete = {
        scene_id: bands
        for scene_id, bands in scenes.items()
        if not BANDS_REQUIRED.issubset(bands.keys())
    }

    print(f"Triplette complete trovate: {len(complete_triplets)}")
    print(f"Scene incomplete ignorate: {len(incomplete)}")
    print(f"Scene riconosciute: {list(complete_triplets.keys())}") # Added print statement

    return complete_triplets


def align_bands(b4, b5, b6):
    """
    Se B4 ha risoluzione doppia rispetto a B5/B6, viene binnata 2x2.
    Poi tutte le bande vengono tagliate alla dimensione comune minima.
    """

    if b4.shape[0] >= 2 * b5.shape[0] - 2 and b4.shape[1] >= 2 * b5.shape[1] - 2:
        print("B4 ha risoluzione doppia: applico binning 2x2")
        b4 = binning2x2_uint16(b4)

    min_h = min(b4.shape[0], b5.shape[0], b6.shape[0])
    min_w = min(b4.shape[1], b5.shape[1], b6.shape[1])

    b4 = b4[:min_h, :min_w]
    b5 = b5[:min_h, :min_w]
    b6 = b6[:min_h, :min_w]

    return b4, b5, b6


def save_cube_patches(scene_id, b4, b5, b6, output_folder):
    """
    Crea cubi a 3 canali e salva patch 512x512.
    Ordine canali: B4, B5, B6.
    """

    os.makedirs(output_folder, exist_ok=True)

    h, w = b4.shape
    patch_index = 0

    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):

            patch_b4 = b4[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_b5 = b5[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_b6 = b6[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            cube = np.stack([patch_b5, patch_b6, patch_b4], axis=-1)

            output_name = f"RGB_{scene_id}_{patch_index:04d}.tif"
            output_path = os.path.join(output_folder, output_name)

            success = cv2.imwrite(output_path, cube)

            if not success:
                print(f"Errore nel salvataggio: {output_path}")
            else:
              test = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
              if test is None or test.dtype != np.uint16 or test.shape != cube.shape:
                print(f"Attenzione: salvataggio non verificato correttamente: {output_path}")


            patch_index += 1

    print(f"{scene_id}: salvate {patch_index} patch")


def process_dataset(input_folder, output_folder):
    triplets = group_triplets(input_folder)

    for scene_id, bands in triplets.items():
        print(f"\nElaboro scena: {scene_id}")

        b4 = read_12bit_image(bands["B4"])
        b5 = read_12bit_image(bands["B5"])
        b6 = read_12bit_image(bands["B6"])

        print("Shape originali:")
        print("B4:", b4.shape, b4.dtype)
        print("B5:", b5.shape, b5.dtype)
        print("B6:", b6.shape, b6.dtype)

        b4, b5, b6 = align_bands(b4, b5, b6)

        print("Shape allineate:")
        print("B4:", b4.shape)
        print("B5:", b5.shape)
        print("B6:", b6.shape)

        save_cube_patches(scene_id, b4, b5, b6, output_folder)


def main():
    input_folder = DATASET_DIR
    output_folder = RGB_FOLDER

    # print(f"Input folder: {input_folder}")
    # print(f"Output folder: {output_folder}")
    process_dataset(input_folder, output_folder)


if __name__ == "__main__":
    main()