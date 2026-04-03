import numpy
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/camilla/Scrivania/Tesi/RGB/RGB_038036_0288.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img1 = cv2.imread('/home/camilla/Scrivania/risultati/img_binn.png', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

img2 = cv2.imread('input2.jpg', cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

img3 = cv2.imread('pre2.jpg', cv2.IMREAD_COLOR)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

fig, axs = plt.subplots(1, 4, figsize=(10, 5))


axs[0].imshow(img[200:400, 155:355], cmap='viridis', vmin=30, vmax=60)
axs[0].set_title('Reference high resolution')
axs[0].axis('off')

axs[1].imshow(img1[100:200, 77:177])
axs[1].set_title('Input RGB')
axs[1].axis('off')

axs[2].imshow(img2[200:400, 155:355], cmap='viridis', vmin=30, vmax=60)
axs[2].set_title('Bicubic')
axs[2].axis('off')

axs[3].imshow(img3[200:400, 155:355], cmap='viridis', vmin=30, vmax=60)
axs[3].set_title('SRCNN')
axs[3].axis('off')

# Layout compatto
plt.tight_layout()
#plt.show()

# Setup figura
fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 

# Lista immagini e titoli
images = [img, img1, img2, img3]
titles = ['Reference high resolution', 'Input RGB', 'Bicubic', 'SRCNN']

# Coordinate crop definite
crops = [(250, 350, 200, 300), (125, 175, 100, 150), (250, 350, 200, 300), (250, 350, 200, 300)]

# Loop su immagini
for i, (image, title, (y1, y2, x1, x2)) in enumerate(zip(images, titles, crops)):
    # Decide se usare viridis o no
    if i == 1:
        axs[i].imshow(image)  # Seconda immagine: scala di grigi
    else:
        axs[i].imshow(image, cmap='viridis', vmin=30, vmax=60)
    axs[i].set_title(title)
    axs[i].axis('off')
    
    # Disegna il rettangolo sull'immagine
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='white', facecolor='none', linewidth=2)
    axs[i].add_patch(rect)

    # Inserisci zoom in basso a sinistra
    inset_size = 0.3  # Percentuale del grafico
    ax_inset = axs[i].inset_axes([0.05, 0.05, inset_size, inset_size])
    zoom_crop = image[y1:y2, x1:x2]
    if i == 1:
        ax_inset.imshow(zoom_crop)  # Secondo zoom: scala di grigi
    else:
        ax_inset.imshow(zoom_crop, cmap='viridis', vmin=30, vmax=60)
    ax_inset.axis('off')

    # Disegna anche un bordo bianco attorno alla miniatura
    ax_inset.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax_inset.transAxes,  # rettangolo grande quanto tutta la miniatura
        edgecolor='white', facecolor='none', linewidth=3
    ))

# Layout compatto
plt.tight_layout()
#plt.show()

input = cv2.imread('/home/camilla/Scrivania/Tesi/RGB/RGB_038036_0288.png',cv2.IMREAD_COLOR)
output = cv2.imread('pre2.jpg', cv2.IMREAD_COLOR)

# Calcola il rapporto per ogni piano (B, G, R) - OpenCV usa BGR
ratio_b = input[:, :, 0] / (output[:, :, 0])
ratio_g = input[:, :, 1] / (output[:, :, 1])
ratio_r = input[:, :, 2] / (output[:, :, 2])

# Visualizza
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Canale R
axs[0].imshow(ratio_r)
axs[0].set_title('Ratio - Red Channel')
axs[0].axis('off')
fig.colorbar(axs[0].imshow(ratio_r))

# Canale G
axs[1].imshow(ratio_g)
axs[1].set_title('Ratio - Green Channel')
axs[1].axis('off')
fig.colorbar(axs[1].imshow(ratio_g))

# Canale B
axs[2].imshow(ratio_b)
axs[2].set_title('Ratio - Blue Channel')
axs[2].axis('off')
fig.colorbar(axs[2].imshow(ratio_b))

plt.tight_layout()
#plt.show()


# Crea una figura con 3 sottoplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Istogramma Red
axs[0].hist(ratio_r.flatten(), bins=50, color='red')
axs[0].set_title('Red Channel')
axs[0].set_xlabel('Ratio Value')
axs[0].set_ylabel('Number of Pixels')
axs[0].grid(True)

# Istogramma Green
axs[1].hist(ratio_g.flatten(), bins=50, color='green')
axs[1].set_title('Green Channel')
axs[1].set_xlabel('Ratio Value')
axs[1].set_ylabel('Number of Pixels')
axs[1].grid(True)

# Istogramma Blue
axs[2].hist(ratio_b.flatten(), bins=50, color='blue')
axs[2].set_title('Blue Channel')
axs[2].set_xlabel('Ratio Value')
axs[2].set_ylabel('Number of Pixels')
axs[2].grid(True)


plt.tight_layout()
plt.show()


# Calcolo statistiche
mean_r, std_r = np.mean(ratio_r), np.std(ratio_r)
mean_g, std_g = np.mean(ratio_g), np.std(ratio_g)
mean_b, std_b = np.mean(ratio_b), np.std(ratio_b)

# Stampa i risultati
print("Ratio statistics:")
print(f"Red channel   - Mean: {mean_r:.4f}, Std Dev: {std_r:.4f}")
print(f"Green channel - Mean: {mean_g:.4f}, Std Dev: {std_g:.4f}")
print(f"Blue channel  - Mean: {mean_b:.4f}, Std Dev: {std_b:.4f}")
