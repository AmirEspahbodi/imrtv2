import os
from PIL import Image, ImageOps
from tqdm import tqdm

def process_image(input_path, output_path):
    """
    Opens an image, adds a white background to make it square, and saves it.

    The original image is centered on the new white canvas. The size of the
    canvas is determined by the larger dimension (width or height) of the
    original image.

    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the processed image file.
    """
    try:
        # Open the original image
        with Image.open(input_path) as img:
            # Correct image orientation based on EXIF data before any processing
            img = ImageOps.exif_transpose(img)

            # Convert image to RGB to handle various modes like RGBA, P, etc.
            # This ensures consistency in the output images.
            img = img.convert('RGB')
            width, height = img.size

            # --- Quality Preservation ---
            # Define save options to maintain high quality, especially for JPEGs.
            save_opts = {}
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                # For JPEGs, quality=95 is a good balance of quality and file size.
                # subsampling=0 preserves color information.
                save_opts = {'quality': 95, 'subsampling': 0}
            # --------------------------

            # If the image is already square, just save a copy and return
            if width == height:
                img.save(output_path, **save_opts)
                return

            # Determine the larger dimension
            max_dim = max(width, height)

            # Create a new white background image
            # The size is square (max_dim x max_dim)
            # The color is white (255, 255, 255)
            new_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))

            # Calculate the position to paste the original image so it's centered
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2

            # Paste the original image onto the white background
            new_img.paste(img, (paste_x, paste_y))

            # Save the newly created square image
            new_img.save(output_path, **save_opts)

    except Exception as e:
        print(f"Error processing image {input_path}: {e}")

def process_dataset(input_folder, output_folder):
    """
    Processes an entire dataset of images organized in class folders.

    It replicates the directory structure of the input folder in the output
    folder and saves the processed, squared images there.

    Args:
        input_folder (str): The path to the root dataset folder.
        output_folder (str): The path to the folder where the cleaned dataset will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    for stage in ("test", ):
        # List all class directories in the input folder
        working_folder = f"{input_folder}"
        if not os.path.exists(f"{output_folder}"):
            os.makedirs(f"{output_folder}")
            print(f"Created directory: {output_folder}")
        class_dirs = [d for d in os.listdir(working_folder) if os.path.isdir(os.path.join(working_folder, d))]

        print(class_dirs)
        
        if not class_dirs:
            print(f"No class directories found in {working_folder}. Exiting.")
            return

        print(f"Found {len(class_dirs)} classes. Starting processing...")

        # Iterate over each class directory with a progress bar
        for class_name in tqdm(class_dirs, desc="Processing Classes"):
            input_class_path = os.path.join(working_folder, class_name)
            output_class_path = os.path.join(f"{output_folder}/{stage}", class_name)

            # Create corresponding class directory in the output folder
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            # List all image files in the current class directory
            image_files = [
                f for f in os.listdir(input_class_path)
                if os.path.isfile(os.path.join(input_class_path, f))
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
            ]

            # Process each image in the class directory
            for image_name in image_files:
                input_image_path = os.path.join(input_class_path, image_name)
                output_image_path = os.path.join(output_class_path, image_name)

                process_image(input_image_path, output_image_path)

    print("\nDataset processing complete.")
    print(f"Cleaned dataset saved in: {output_folder}")

if __name__ == '__main__':
    # --- Configuration ---
    # Set the name of your original dataset folder
    INPUT_DATASET_FOLDER = '../car_dataset_original'
    # Set the name for the new, cleaned dataset folder
    OUTPUT_DATASET_FOLDER = './same_sized_car_dataset'
    # -------------------

    # Check if the input directory exists before starting
    if not os.path.isdir(INPUT_DATASET_FOLDER):
        print(f"Error: Input directory '{INPUT_DATASET_FOLDER}' not found.")
        print("Please make sure your dataset is in a folder named 'dataset' or change the INPUT_DATASET_FOLDER variable.")
    else:
        process_dataset(INPUT_DATASET_FOLDER, OUTPUT_DATASET_FOLDER)

