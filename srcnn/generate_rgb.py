import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import re
    
def extract_sub_matrices(image, block_dim):
    n,m = image.shape[:2] 
    sub_matrices = []
    
    for i in range(0, n - block_dim + 1, block_dim):
        for j in range(0, m - block_dim + 1, block_dim):
            sub_matrix = image[i:i+block_dim, j: j+block_dim]            
            sub_matrices.append(sub_matrix)
    
    return sub_matrices

def save_images(sub_matrices, output_folder, start_index, prefix, image_filename):

    #Extract triplet number
    base_name = os.path.basename(image_filename)
    match = re.search(r"LC08_L1[TG][TP]_(\d{6})_", base_name)    
    id_part = match.group(1)

    for idx, sub_matrix in enumerate(sub_matrices):
        #filename = os.path.join(output_folder, f"{prefix}_{start_index + idx:04d}.png")
        filename = os.path.join(output_folder, f"{prefix}_{id_part}_{start_index + idx:04d}.png")
        success = cv2.imwrite(filename, sub_matrix) 
        if not success:
            print(f"Error on saving {filename}")
    
    return start_index + len(sub_matrices)

def extract_512x512(images, output_folder, prefix):
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  

    # set = extract_sub_matrices(image, 512)
    # print("len set: ", len(set))
    # current_index = save_images(set, output_folder, 0, prefix)

    # Find images
    i = 0
    current_index = 0
    for img_path in images:
        # Load the image
        image = cv2.imread(img_path)

        size = image.shape[:2]
        print(f"Image {i} size: ", size)
        i += 1

        set = extract_sub_matrices(image, 512)
        print("len set: ", len(set))
        current_index = save_images(set, output_folder, current_index, prefix, img_path)

def binning2x2(image, offset_x, offset_y):
   
    n,m = image.shape[:2]

    new_height = (n - offset_y) // 2 + (1 if (n - offset_y) % 2 != 0 else 0)
    new_width = (m -offset_x) // 2 + (1 if (m - offset_x) % 2 != 0 else 0)

    low_img = np.zeros((new_height, new_width))
    
    for i in range(0, n, 2):
        for j in range(0, m, 2):
            
            #block with offset
            i_offset = i + offset_y
            j_offset = j + offset_x
            if i_offset < n and j_offset < m:
                block = image[i_offset:min(i_offset+2, n), j_offset:min(j_offset+2, m)]
                low_img[i//2, j//2] = np.mean(block)

    return low_img

def extract_256x256(images, output_folder, prefix, offset_x, offset_y):

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  

    # low = binning2x2(image, offset_x, offset_y)
    # print("len low: ", low.shape[:2])

    # set = extract_sub_matrices(low, 256)
    # print("len set: ", len(set))
    # current_index = save_images(set, output_folder, 0, prefix)

    # Find images
    i = 0
    current_index = 0
    for img_path in images:
        # Load the image
        image = cv2.imread(img_path)

        size = image.shape[:2]
        print(f"Image {i} size: ", size)
        i += 1

        low = binning2x2(image, offset_x, offset_y)
        print("len low: ", low.shape[:2])

        set = extract_sub_matrices(low, 256)
        print("len set: ", len(set))
        current_index = save_images(set, output_folder, current_index, prefix, img_path)

def extract_512x512_B8(images, output_folder, prefix):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  

    # image = image[1:, 0:]
    # print("Cut B8 size: ", image.shape[:2])

    # low = binning2x2(image, 0, 0)
    # print("len low: ", low.shape[:2])

    # set = extract_sub_matrices(low, 512)
    # print("len set: ", len(set))
    # current_index = save_images(set, output_folder, 0, prefix)


    # Find images
    i = 0
    current_index = 0
    for img_path in images:
        # Load the image
        image = cv2.imread(img_path)

        size = image.shape[:2]
        print(f"Image {i} size: ", size)
        i += 1

        image_cut = image[1:, 0:]
        print("Cut B8 size: ", image_cut.shape[:2])

        low = binning2x2(image_cut, 0, 0)
        print("len low: ", low.shape[:2])

        set = extract_sub_matrices(low, 512)
        print("len set: ", len(set))
        current_index = save_images(set, output_folder, current_index, prefix,img_path)

def extract_256x256_B8(images, output_folder, prefix):

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)  

    # low1 = binning2x2(image,0,0)
    # low2 = binning2x2(image, 1, 1)
    # print("len low1: ", low1.shape[:2])
    # print("len low2: ", low2.shape[:2])


    # set1 = extract_sub_matrices(low1, 256)
    # print("len set1: ", len(set1))
    # current_index = save_images(set1, output_folder, 0, prefix)

    # set2 = extract_sub_matrices(low2, 256)
    # print("len set2: ", len(set2))
    # current_index = save_images(set2, output_folder, current_index, prefix)

    # Find images
    i = 0
    current_index = 0
    for img_path in images:
        # Load the image
        image = cv2.imread(img_path)

        size = image.shape[:2]
        print(f"Image {i} size: ", size)
        i += 1

        image_cut = image[1:, 0:]
        print("Cut B8 size: ", image_cut.shape[:2])

        low1 = binning2x2(image_cut,0,0)
        low2 = binning2x2(image_cut, 1, 1)
        print("len low1: ", low1.shape[:2])
        print("len low2: ", low2.shape[:2])

        set1 = extract_sub_matrices(low1, 256)
        print("len set1: ", len(set1))
        current_index = save_images(set1, output_folder, current_index, prefix, img_path)

        set2 = extract_sub_matrices(low2, 256)
        print("len set2: ", len(set2))
        current_index = save_images(set2, output_folder, current_index, prefix, img_path)


def crea_cubo_rgb_unico(folder_B5, folder_B8, folder_B4, output_folder, id_noB8):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_B8):
        match = re.match(r"B8_(\d{6}_\d{4})\.png", filename)
        if not match:
            continue

        id = match.group(1)

        path_B8 = os.path.join(folder_B8, filename)
        path_B4 = os.path.join(folder_B4, f"B4_{id}.png")
        path_B5 = os.path.join(folder_B5, f"B5_{id}.png")

        if not (os.path.exists(path_B4) and os.path.exists(path_B5)):
            print(f"Mancano B4 o B5 per ID {id}")
            continue

        # Leggi immagini
        b4 = cv2.imread(path_B4, cv2.IMREAD_GRAYSCALE)
        b5 = cv2.imread(path_B5, cv2.IMREAD_GRAYSCALE)
        b8 = cv2.imread(path_B8, cv2.IMREAD_GRAYSCALE)

        if b5 is None or b8 is None or b4 is None:
            raise ValueError("One or more images were not loaded correctly.")


        # Stack on 3 channels: R = B5, G = B8, B = B4
        cube_rgb = np.stack([b5, b8, b4], axis=-1)
        nome_output = os.path.join(f"RGB_{id}.png")


        # Salva l'immagine
        output_path = os.path.join(output_folder, nome_output)
        cv2.imwrite(output_path, cube_rgb)

    




def main():
    
    # folder of images
    folder = "/home/camilla/Scrivania/Tesi/Images"
    
    ##-----------B4 -B5----------------------------------

    # Output folder where sub-images will be saved
    output_folder_B4_1 = "/home/camilla/Scrivania/Tesi/Data_Set_B4_512"
    output_folder_B4_2 = "/home/camilla/Scrivania/Tesi/Data_Set_B4_256"

    # Load the image B4
    # image_B4 = cv2.imread("/home/camilla/Scrivania/Tesi/Images/LC08_L1TP_193028_20210221_20210303_02_T1_B4.TIF")
    # size = image_B4.shape[:2]
    # print("Image B4 size: ", size)

    images_B4 = sorted(glob.glob(os.path.join(folder, "*B4*.TIF")))
    print("Number of images found: ", len(images_B4))

    #extract_512x512(images_B4,output_folder_B4_1,"B4")
    #extract_256x256(images_B4,output_folder_B4_2, "B4",0,0)


    # Output folder where sub-images will be saved
    output_folder_B5_1 = "/home/camilla/Scrivania/Tesi/Data_Set_B5_512"
    output_folder_B5_2 = "/home/camilla/Scrivania/Tesi/Data_Set_B5_256"


    # Load the image B5
    # image_B5 = cv2.imread("/home/camilla/Scrivania/Tesi/Images/LC08_L1TP_193028_20210221_20210303_02_T1_B5.TIF")
    # size = image_B5.shape[:2]
    # print("Image B5 size: ", size)
    images_B5 = sorted(glob.glob(os.path.join(folder, "*B5*.TIF")))
    print("Number of images found: ", len(images_B5))

    #extract_512x512(images_B5,output_folder_B5_1,"B5")
    #extract_256x256(images_B5,output_folder_B5_2, "B5",1,1)


    ##--------------B8-------------------------------------------
    # Load the image B8
    # image_B8 = cv2.imread("/home/camilla/Scrivania/Tesi/Images/LC08_L1TP_193028_20210221_20210303_02_T1_B8.TIF")
    # print("Original B8 size: ", image_B8.shape[:2])
   
    images_B8 = sorted(glob.glob(os.path.join(folder, "*B8*.TIF")))
    print("Number of images found: ", len(images_B8))

    output_folder_B8_1 = "/home/camilla/Scrivania/Tesi/Data_Set_B8_512"
    #extract_512x512_B8(images_B8, output_folder_B8_1, "B8")   

    output_folder_B8_2 = "/home/camilla/Scrivania/Tesi/Data_Set_B8_256"
    #extract_256x256_B8(images_B8, output_folder_B8_2, "B8")


    ##---------------------------RGB---------------------------------
    no_B8 = "192030_20190516"
    base_folder = "/home/camilla/Scrivania/Tesi"
    folder_B4 = os.path.join(base_folder, "Data_Set_B4_512")
    folder_B5 = os.path.join(base_folder, "Data_Set_B5_512")
    folder_B8 = os.path.join(base_folder, "Data_Set_B8_512")
    output_folder_RGB = os.path.join(base_folder, "RGB")

    # id = "038036_0288"
    # # Genera i path dei file basati sull'identificativo
    # B4_path = os.path.join(folder_B4, f"B4_{id}.png")
    # B5_path = os.path.join(folder_B5, f"B5_{id}.png")
    # B8_path = os.path.join(folder_B8, f"B8_{id}.png")

    crea_cubo_rgb_unico(folder_B5, folder_B8, folder_B4, output_folder_RGB, no_B8)

    # b5 = cv2.imread(B5_path, cv2.IMREAD_GRAYSCALE)
    # b8 = cv2.imread(B8_path, cv2.IMREAD_GRAYSCALE)
    # b4 = cv2.imread(B4_path, cv2.IMREAD_GRAYSCALE)
 

    # # Salva con ID nel nome
    # cv2.imwrite(os.path.join("/home/camilla/Scrivania/Tesi/RGB", f"B4_{id}.png"), b4)
    # cv2.imwrite(os.path.join("/home/camilla/Scrivania/Tesi/RGB", f"B5_{id}.png"), b5)
    # cv2.imwrite(os.path.join("/home/camilla/Scrivania/Tesi/RGB", f"B8_{id}.png"), b8)

if __name__ == "__main__":
    main()