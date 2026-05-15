from pathlib import Path
import sys
import numpy
import cv2

TIF_B4 = Path("LC08_L1TP_188028_20210218_20210302_02_T1_B4.TIF")
TIF_B5 = Path("LC08_L1TP_188028_20210218_20210302_02_T1_B5.TIF")
TIF_B8 = Path("LC08_L1TP_188028_20210218_20210302_02_T1_B8.TIF")


def binning2x2(image, offset_x, offset_y):
   
    n,m = image.shape[:2]

    new_height = (n - offset_y) // 2 + (1 if (n - offset_y) % 2 != 0 else 0)
    new_width = (m -offset_x) // 2 + (1 if (m - offset_x) % 2 != 0 else 0)

    low_img = numpy.zeros((new_height, new_width))
    
    for i in range(0, n, 2):
        for j in range(0, m, 2):
            
            #block with offset
            i_offset = i + offset_y
            j_offset = j + offset_x
            if i_offset < n and j_offset < m:
                block = image[i_offset:min(i_offset+2, n), j_offset:min(j_offset+2, m)]
                low_img[i//2, j//2] = numpy.mean(block)

    return low_img

def main() -> int:
    B4 = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else TIF_B4
    B5 = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else TIF_B5
    B8 = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else TIF_B8

    image_B4 = cv2.imread(str(B4), cv2.IMREAD_UNCHANGED)
    image_B5 = cv2.imread(str(B5), cv2.IMREAD_UNCHANGED)
    image_B8 = cv2.imread(str(B8), cv2.IMREAD_UNCHANGED)


    if image_B4 is None or image_B5 is None or image_B8 is None:
        print("Errore nel caricamento di uno o più file.")
        return 1
    
    image_B4 = image_B4.astype(numpy.float32) / 16.0
    image_B5 = image_B5.astype(numpy.float32) / 16.0
    image_B8 = image_B8.astype(numpy.float32) / 16.0

    print(f"File: {B5.name}")
    print(f"Shape: {image_B5.shape}")
    print(f"Dtype: {image_B5.dtype}")
    print("min:", numpy.min(image_B5))
    print("max:", numpy.max(image_B5))
    print("unique sample:", numpy.unique(image_B5[:100, :100])[:20])
    print("\n")

    print(f"File: {B4.name}")
    print(f"Shape: {image_B4.shape}")
    print(f"Dtype: {image_B4.dtype}")
    print("min:", numpy.min(image_B4))
    print("max:", numpy.max(image_B4))
    print("unique sample:", numpy.unique(image_B4[:100, :100])[:20])
    print("\n")

    print(f"File: {B8.name}")
    print(f"Shape: {image_B8.shape}")
    print(f"Dtype: {image_B8.dtype}")
    print("min:", numpy.min(image_B8))
    print("max:", numpy.max(image_B8))
    print("unique sample:", numpy.unique(image_B8[:100, :100])[:20])
    print("\n")

    binned_B8 = binning2x2(image_B8, 0, 0)
    print(f"Binned B8 shape: {binned_B8.shape}")

    cube_rgb = numpy.stack([image_B5, binned_B8, image_B4], axis=-1)
    print(f"Cube RGB shape: {cube_rgb.shape}")  

    # Salva la reference
    reference = numpy.clip(cube_rgb, 0, 4095).astype(numpy.uint16)
    cv2.imwrite("reference.tif", reference)

    # Aggiunta rumore gaussiano
    numpy.random.seed(42)
    sigma = 50.0
    noise = numpy.random.normal(0.0, sigma, cube_rgb.shape).astype(numpy.float32)
    noisy_rgb = cube_rgb + noise

    # Clipping e salvataggio immagine degradata
    noisy_rgb = numpy.clip(noisy_rgb, 0, 4095).astype(numpy.uint16)
    cv2.imwrite("input.tif", noisy_rgb)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
