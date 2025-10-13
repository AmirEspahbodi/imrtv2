import cv2
import os
import glob
import albumentations as A
import numpy as np
from tqdm import tqdm



def create_augmentation_pipeline():
    """
    Defines and returns the Albumentations pipeline.
    The transformations are chosen to be subtle and realistic for e-commerce
    image retrieval, preserving key product features, details, and colors.
    The parameters have been reduced for less aggressive augmentation.
    """
    return A.Compose([
        # --- Geometric Transformations ---

        # Apply slight rotation, scaling, and shifting.
        # The background is filled with white (255, 255, 255) to match the
        # original product images.
        A.ShiftScaleRotate(
            shift_limit=0.08,      # Reduced from 0.08
            scale_limit=0.1,      # Reduced from 0.1
            rotate_limit=15,       # Reduced from 15
            p=0.8,                 # Reduced from 0.8
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        ),

        # Apply a slight perspective transformation.
        A.Perspective(
            scale=(0.01, 0.04),    # Reduced from (0.02, 0.06)
            p=0.3,                 # Reduced from 0.4
            pad_val=(255, 255, 255) # Pad with white background
        ),

        # Horizontal flip is a very common and effective augmentation.
        A.HorizontalFlip(p=0.5), # 50% chance of applying (unchanged)

        # --- Photometric (Quality & Color) Transformations ---

        # Adjust brightness and contrast to simulate different lighting conditions.
        A.RandomBrightnessContrast(
            brightness_limit=0.1, # Reduced from 0.15
            contrast_limit=0.1,   # Reduced from 0.15
            p=0.6                 # Reduced from 0.7
        ),

        # Apply one of the following blurring/noise effects to simulate
        # lower-quality user photos (e.g., slight motion or focus issues).
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5), # Max blur limit reduced from 7
            A.MotionBlur(blur_limit=(3, 5), p=0.5), # Max blur limit reduced from 7
        ], p=0.4), # Reduced from 0.5

        # Add Gaussian noise to simulate camera sensor noise.
        A.GaussNoise(var_limit=(7.0, 35.0), p=0.3), # Reduced from (10.0, 50.0) and p=0.3

        # Very subtle color shifting. Kept low to not change product color identity.
        A.HueSaturationValue(
            hue_shift_limit=3,     # Reduced from 5
            sat_shift_limit=10,    # Reduced from 15
            val_shift_limit=10,    # Reduced from 15
            p=0.3                  # Reduced from 0.4
        ),
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
    if not os.path.exists(root_output_dir):
        os.makedirs(root_output_dir)
        print(f"Created root output directory: {root_output_dir}")

    # Find all subdirectories in the input directory (these are the classes)
    class_dirs = [d for d in os.listdir(root_input_dir) if os.path.isdir(os.path.join(root_input_dir, d))]

    if not class_dirs:
        print(f"No class directories found in {root_input_dir}. Please check the structure.")
        return

    print(f"Found {len(class_dirs)} class directories.")
    total_images_generated = 0
    pipeline = create_augmentation_pipeline()

    # Process each class directory
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_input_path = os.path.join(root_input_dir, class_name)
        class_output_path = os.path.join(root_output_dir, class_name)

        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        # Find all image files in the current class directory
        image_paths = glob.glob(os.path.join(class_input_path, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(class_input_path, '*.[jJ][pP][eE][gG]')) + \
                      glob.glob(os.path.join(class_input_path, '*.[pP][nN][gG]'))

        if not image_paths:
            continue

        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                base_filename = os.path.splitext(os.path.basename(img_path))[0]

                for i in range(num_augmentations_per_image):
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
    INPUT_IMAGE_DIRECTORY = "dataset"
    OUTPUT_IMAGE_DIRECTORY = "augmented_dataset"
    IMAGES_PER_ORIGINAL = 25 # Generate 25 new images for each original

    # Create a dummy input directory and a fake image for demonstration
    if not os.path.exists(INPUT_IMAGE_DIRECTORY):
        print("Creating dummy dataset structure for demonstration.")
        dummy_class_dir = os.path.join(INPUT_IMAGE_DIRECTORY, "dummy_class_12345")
        os.makedirs(dummy_class_dir)
        dummy_image = np.full((512, 512, 3), (200, 220, 240), dtype=np.uint8)
        cv2.putText(dummy_image, 'Product', (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        sample_image_path = os.path.join(dummy_class_dir, "sample_product.png")
        cv2.imwrite(sample_image_path, dummy_image)
        print(f"A sample image has been created at: {sample_image_path}")


    # --- Run the Pipeline ---
    process_images(
        root_input_dir=INPUT_IMAGE_DIRECTORY,
        root_output_dir=OUTPUT_IMAGE_DIRECTORY,
        num_augmentations_per_image=IMAGES_PER_ORIGINAL
    )