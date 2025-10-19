import os
from PIL import Image
from tqdm import tqdm # For a user-friendly progress bar

def process_dataset(source_folder, dest_folder, target_size=(512, 512)):
    """
    Copies an entire image dataset folder structure from source to destination,
    resizing all images in the process.

    Args:
        source_folder (str): The path to the source dataset directory.
        dest_folder (str): The path to the destination directory for resized images.
        target_size (tuple): A tuple (width, height) for the output image size.
    """
    # Find all image files to process to set up the progress bar
    image_paths = []
    supported_formats = ('.jpg', '.jpeg', '.png')
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"Warning: No images found in '{source_folder}'.")
        return

    print(f"Found {len(image_paths)} images. Starting processing...")

    # Process each image with a progress bar
    for src_path in tqdm(image_paths, desc="Resizing and Copying"):
        try:
            # --- This is the core logic for preserving folder structure ---
            # 1. Get the path of the image relative to the source folder.
            #    Example: 'product_category/item_123/front_view.jpg'
            relative_path = os.path.relpath(src_path, source_folder)

            # 2. Create the full destination path for the new image.
            #    Example: 'resized_dataset/product_category/item_123/front_view.jpg'
            dest_path = os.path.join(dest_folder, relative_path)
            
            # 3. Create the destination subdirectories if they don't exist.
            #    Example: 'resized_dataset/product_category/item_123/'
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # ----------------------------------------------------------------

            # Open, resize, and save the image
            with Image.open(src_path) as img:
                # Use LANCZOS for high-quality downsampling. It's one of the best.
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB to ensure a 3-channel image (handles PNGs with transparency)
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                
                # Save the new image with high quality
                img_resized.save(dest_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"\nCould not process file {src_path}. Error: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # ❗️ Set your paths here
    source_dataset_path = "D:\\amir_es\\car_accessories_dataset_splited"
    resized_dataset_path = "D:\\amir_es\\car_accessories_dataset_downsized" # This will be created in your current directory

    process_dataset(source_dataset_path, resized_dataset_path)

    print(f"✅ Processing complete. Resized dataset is available at: '{resized_dataset_path}'")