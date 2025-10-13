import os
from PIL import Image
from tqdm import tqdm # A library for a smart progress bar

def resize_images_in_directory(source_dir, dest_dir, target_size=(512, 512)):
    """
    Recursively finds all images in the source directory, resizes them,
    and saves them to the destination directory, preserving the folder structure.

    Args:
        source_dir (str): The path to the root directory of the source images.
        dest_dir (str): The path to the root directory where resized images will be saved.
        target_size (tuple): A tuple (width, height) for the output image size.
    """
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png')
    
    # Create a list of all image files to process for the progress bar
    image_files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files_to_process.append(os.path.join(root, file))

    if not image_files_to_process:
        print("No images found in the source directory.")
        return

    print(f"Found {len(image_files_to_process)} images to resize.")

    # Process images with a progress bar
    for src_path in tqdm(image_files_to_process, desc="Resizing Images"):
        try:
            # Construct the destination path by replacing the source base with the destination base
            relative_path = os.path.relpath(src_path, source_dir)
            dest_path = os.path.join(dest_dir, relative_path)
            
            # Create the destination subdirectory if it doesn't exist
            dest_folder = os.path.dirname(dest_path)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Open, resize, and save the image
            with Image.open(src_path) as img:
                # Use LANCZOS for high-quality downsampling
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB to handle formats like RGBA or P (palette)
                # This ensures the output is always 3 channels
                img_resized = img_resized.convert('RGB')
                
                # Save the resized image
                img_resized.save(dest_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"Error processing {src_path}: {e}")

# --- Configuration ---
if __name__ == "__main__":
    # ❗️ IMPORTANT: Set your source and destination paths here
    source_directory = "E:\\imrtv2\\augmented_car_accessories_dataset"
    destination_directory = "E:\\imrtv2\\resized_augmented_car_accessories_dataset"
    
    print("Starting image resizing process...")
    resize_images_in_directory(source_directory, destination_directory)
    print("✅ Image resizing complete!")