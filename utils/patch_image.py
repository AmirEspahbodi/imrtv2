import cv2
import os
import glob
from tqdm import tqdm
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "E:\imrtv2\\testing\\102245"
OUTPUT_DIR = "E:\imrtv2\\testing\\result"
TARGET_SIZE = 512
# Threshold to distinguish product from white background (0-255).
# Lower this value if the background is grayish.
BACKGROUND_THRESHOLD = 245

BACKGROUND_COVERAGE_THRESHOLD = 0.70

# --- SCRIPT ---

def create_spatial_augmentations(image_path, output_folder):
    """
    Detects a product in an image, slides a window over it, validates the content,
    and saves the patches.

    Args:
        image_path (str): The full path to the input image.
        output_folder (str): The folder where augmented patches will be saved.
    """
    try:
        # 1. Load the image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # 2. Detect the edge of the product (ROI detection)
        _, thresh = cv2.threshold(gray_image, BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found in {image_path}. Skipping.")
            return

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        if w < 20 or h < 20:
            print(f"Warning: Detected product is too small in {image_path}. Skipping.")
            return

        # 3. Define window and stride size
        product_dim = min(w, h)
        window_size = int(product_dim / 4)
        stride = int(product_dim / 6)

        if window_size < 1 or stride < 1:
            print(f"Warning: Calculated window/stride is zero for {image_path}. Skipping.")
            return

        patch_count = 0
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 4. Start sliding the window
        for y_pos in range(y, y + h, stride):
            for x_pos in range(x, x + w, stride):
                if (y_pos + window_size > y + h) or (x_pos + window_size > x + w):
                    continue

                patch = original_image[y_pos:y_pos + window_size, x_pos:x_pos + window_size]
                
                if patch.size == 0:
                    continue

                resized_patch = cv2.resize(patch, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

                # 5. FINAL CHECK: Discard patches with too much background
                gray_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)
                white_pixel_count = np.sum(gray_patch > BACKGROUND_THRESHOLD)
                total_pixels = TARGET_SIZE * TARGET_SIZE
                background_percentage = white_pixel_count / total_pixels

                # If the patch is mostly background, skip saving it
                if background_percentage > BACKGROUND_COVERAGE_THRESHOLD:
                    continue

                # 6. Save the valid patch
                output_filename = f"{base_filename}_patch_{y_pos}_{x_pos}.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, resized_patch)
                patch_count += 1

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")



def main():
    """
    Main function to run the augmentation pipeline.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all images in the input directory
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.jpeg')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.png'))

    if not image_paths:
        print(f"Error: No images found in the '{INPUT_DIR}' directory.")
        print("Please create the folder and add your product images.")
        return

    print(f"Found {len(image_paths)} images. Starting augmentation process...")

    # Process each image with a progress bar
    for path in tqdm(image_paths, desc="Augmenting Images"):
        create_spatial_augmentations(path, OUTPUT_DIR)

    print("\nProcessing complete.")
    print(f"All generated patches have been saved to the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()