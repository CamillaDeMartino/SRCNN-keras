from pathlib import Path

# Paths
DATASET_DIR = Path("/storage/internal_02/SRCNN/dataset")
PROJECT_DIR = Path("/storage/internal_02/SRCNN")
TRAIN_RGB = PROJECT_DIR / "Train_RGB"
TEST_RGB = PROJECT_DIR / "Test_RGB"
PROCESSED_DIR = PROJECT_DIR / "processed"
RGB_FOLDER = PROJECT_DIR / "dataset_rgb" 
RGB_NORM = PROJECT_DIR / "dataset_rgb_norm"

# Training parameters
Random_Crop = 30
Patch_size = 32
label_size = 20
conv_side = 6
scale = 2

# Block params
BLOCK_STEP = 16
BLOCK_SIZE = 32

# File extensions
VALID_EXT = [".png", ".tif", ".tiff", ".TIFF", ".jpg", ".jpeg", ".jp2"]
