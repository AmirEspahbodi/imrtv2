import cv2
import os
import glob
import albumentations as A
import numpy as np
from tqdm import tqdm
import time


def create_augmentation_pipeline():
    return A.Compose([
        # --- Geometric Transformations ---
        # These are still valuable, but we'll apply them slightly less often.
        A.ShiftScaleRotate(
            shift_limit=0.06,      # Slightly reduced for less drastic shifts
            scale_limit=0.08,      # Slightly reduced for less zooming
            rotate_limit=12,       # Reduced rotation angle
            p=0.7,                 # Reduced probability from 0.8
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        ),
        A.HorizontalFlip(p=0.5), # A classic, highly effective augmentation

        # --- Photometric (Quality & Color) Transformations ---
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.6
        ),

        # Grouped blurring and noise effects.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0), 
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.4), 


        A.OneOf([
            A.SaltAndPepper(
                p_noise=(0.001, 0.1),
                p=1.0                
            ),
            A.GaussNoise(
                var_limit=(3.0, 10.0),
                p=1.0                
            )
        ], p=1),
        # Very subtle color shifting to maintain product color integrity.
        A.HueSaturationValue(
            hue_shift_limit=3,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.3
        ),

        # Removed Perspective transform as it can be too distorting for small datasets.
    ])


def process_images(root_input_dir, root_output_dir, num_augmentations_per_image=25):
    """
    Scans for class subdirectories in the root input directory, replicates the
    structure in the output directory, and saves augmented images for each class.

    Args:
        root_input_dir (str): Path to the root directory containing class folders.
        root_output_dir (str): Path to the root directory to save augmented data.
        num_augmentations_per_image (int): The number of augmented versions
                                           to create for each original image.
    """
    for stage in ("train",):
        current_root_output_dir = f"{root_output_dir}"
        current_root_input_dir = f"{root_input_dir}"
        if not os.path.exists(current_root_output_dir):
            os.makedirs(current_root_output_dir)
            print(f"Created root output directory: {current_root_output_dir}")

        # Find all subdirectories in the input directory (these are the classes)
        class_dirs = [d for d in os.listdir(current_root_input_dir) if os.path.isdir(os.path.join(current_root_input_dir, d))]

        if not class_dirs:
            print(f"No class directories found in {current_root_input_dir}. Please check the structure.")
            return

        print(f"Found {len(class_dirs)} class directories.")
        total_images_generated = 0
        pipeline = create_augmentation_pipeline()

        # Process each class directory
        for class_name in tqdm(class_dirs, desc="Processing classes"):
            class_input_path = os.path.join(current_root_input_dir, class_name)
            class_output_path = os.path.join(current_root_output_dir, class_name)

            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)

            # Find all image files in the current class directory
            image_paths = glob.glob(os.path.join(class_input_path, '*.[jJ][pP][gG]')) + \
                        glob.glob(os.path.join(class_input_path, '*.[jJ][pP][eE][gG]')) + \
                        glob.glob(os.path.join(class_input_path, '*.[pP][nN][gG]'))

            if not image_paths:
                continue
            time.sleep(10)
            for img_path in image_paths:
                time.sleep(3)
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    base_filename = os.path.splitext(os.path.basename(img_path))[0]
                    
                    for i in range(num_augmentations_per_image):
                        time.sleep(1)
                        augmented = pipeline(image=image_rgb)
                        augmented_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                        # Create a new, globally unique filename
                        new_filename = f"{class_name}_{base_filename}_aug_{i+1:02d}.jpg"
                        save_path = os.path.join(class_output_path, new_filename)
                        cv2.imwrite(save_path, augmented_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        total_images_generated += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\nAugmentation process complete.")
    print(f"Generated {total_images_generated} new images in {root_output_dir}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Change these paths to your actual directories.
    INPUT_IMAGE_DIRECTORY = "../car_dataset_same_sized"
    OUTPUT_IMAGE_DIRECTORY = "./car_dataset_new_aug"
    IMAGES_PER_ORIGINAL = 6 # Generate 25 new images for each original


    # --- Run the Pipeline ---
    process_images(
        root_input_dir=INPUT_IMAGE_DIRECTORY,
        root_output_dir=OUTPUT_IMAGE_DIRECTORY,
        num_augmentations_per_image=IMAGES_PER_ORIGINAL
    )