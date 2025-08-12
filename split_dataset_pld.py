import os
import shutil
import random
import argparse
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage as ndi
import cv2


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def apply_medical_augmentation(image):
    techniques = ["rotate", "scale", "contrast", "noise", "clahe", "elastic"]
    # Choose between 2 and 4 techniques (allowing repeats)
    selected = random.choices(techniques, k=random.randint(2, 4))

    augmented = image.copy()

    for tech in selected:
        if tech == "rotate":
            angle = random.uniform(-10, 10)
            rotated = augmented.rotate(angle, resample=Image.BICUBIC, expand=True)
            # Fit back to original size
            augmented = ImageOps.fit(
                rotated, image.size, method=Image.BICUBIC, centering=(0.5, 0.5)
            )

        elif tech == "scale":
            scale_factor = random.uniform(0.9, 1.1)
            w, h = augmented.size
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            scaled = augmented.resize((new_w, new_h), resample=Image.BICUBIC)
            # Pad or crop back to original
            pad_w = max(0, w - new_w)
            pad_h = max(0, h - new_h)
            # fill=0 (black) is safer than None to avoid unexpected behavior
            padded = ImageOps.expand(scaled, border=(pad_w // 2, pad_h // 2), fill=0)
            augmented = padded.crop((0, 0, w, h))

        elif tech == "contrast":
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(factor)

        elif tech == "noise":
            # Poisson noise applied on image intensity
            arr = np.array(augmented).astype(np.float32) / 255.0
            noisy = np.random.poisson(arr * 255.0) / 255.0
            noisy_img = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
            augmented = Image.fromarray(noisy_img)

        elif tech == "clahe":
            arr = np.array(augmented)
            # Ensure uint8
            if arr.dtype != np.uint8:
                arr = (
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    if arr.max() <= 1.0
                    else arr.astype(np.uint8)
                )

            # If grayscale or single channel
            if arr.ndim == 2:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(arr)
                augmented = Image.fromarray(cl)
            # If color (apply per-channel)
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                channels = []
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # If 4 channels (RGBA), process RGB and keep alpha as-is
                ccount = arr.shape[2]
                for c in range(min(3, ccount)):
                    channels.append(clahe.apply(arr[:, :, c]))
                if ccount == 4:
                    alpha = arr[:, :, 3]
                    merged = np.stack((*channels, alpha), axis=-1)
                else:
                    merged = np.stack(channels, axis=-1)
                augmented = Image.fromarray(merged)
            else:
                # Unexpected shape, fallback to equalize
                augmented = ImageOps.equalize(augmented)

        elif tech == "elastic":
            # Parameters (conservative by default)
            # If you have no lung mask, prefer the lower end of alpha
            alpha = random.uniform(8.0, 15.0)  # strength of displacement
            sigma = random.uniform(3.0, 6.0)  # smoothness

            arr = np.array(augmented)
            # convert to float for interpolation
            if arr.dtype != np.float32:
                arr_f = arr.astype(np.float32)
            else:
                arr_f = arr

            h, w = arr_f.shape[:2]

            # generate displacement fields
            dx = np.random.rand(h, w) * 2 - 1
            dy = np.random.rand(h, w) * 2 - 1
            dx = ndi.gaussian_filter(dx, sigma=sigma, mode="reflect") * alpha
            dy = ndi.gaussian_filter(dy, sigma=sigma, mode="reflect") * alpha

            # meshgrid and indices
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.vstack(
                (
                    (y + dy).reshape(1, -1),
                    (x + dx).reshape(1, -1),
                )
            )

            # apply to each channel
            if arr_f.ndim == 2:
                warped = ndi.map_coordinates(
                    arr_f, indices, order=1, mode="reflect"
                ).reshape(h, w)
            else:
                channels = []
                for c in range(arr_f.shape[2]):
                    remapped = ndi.map_coordinates(
                        arr_f[..., c], indices, order=1, mode="reflect"
                    ).reshape(h, w)
                    channels.append(remapped)
                warped = np.stack(channels, axis=-1)

            # convert back to uint8 safely
            warped_clipped = np.clip(warped, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(warped_clipped)

        else:
            # Should not happen
            raise RuntimeError(f"Unknown augmentation: {tech}")

    return augmented


def create_directory_structure(destination_root, classes):
    for split in ["train", "validation", "test"]:
        for class_name in classes:
            path = os.path.join(destination_root, split, class_name)
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")


def copy_files_between_splits(
    source_root, split_from, split_to, classes, random_seed=42
):
    random.seed(random_seed)
    total = 0
    for class_name in classes:
        src_dir = os.path.join(source_root, split_from, class_name)
        tgt_dir = os.path.join(source_root, split_to, class_name)
        os.makedirs(tgt_dir, exist_ok=True)
        files = [
            f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))
        ]
        for f in files:
            try:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(tgt_dir, f))
                total += 1
            except Exception as e:
                logger.error(f"Failed to copy {f}: {e}")
    return total


def setup_dataset(original_path, target_path, augment_flag):
    try:
        classes = os.listdir(os.path.join(original_path, "train"))
        logger.info(f"Found classes: {classes}")
    except Exception as e:
        logger.error(f"Unable to read classes: {e}")
        return False

    create_directory_structure(target_path, classes)

    for split in ["train", "test"]:
        for class_name in classes:
            src = os.path.join(original_path, split, class_name)
            dst = os.path.join(target_path, split, class_name)
            os.makedirs(dst, exist_ok=True)
            files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
            logger.info(
                f"Copying {'and augmenting' if split == 'train' and augment_flag == 1 else ''}{len(files)} from {src} to {dst}"
            )
            for f in files:
                src_file = os.path.join(src, f)
                dst_file = os.path.join(dst, f)
                shutil.copy2(src_file, dst_file)

                if split == "train" and augment_flag == 1:
                    name, ext = os.path.splitext(f)
                    try:
                        with Image.open(src_file).convert("L") as img_gray:
                            if class_name == "PNEUMONIA":
                                for i in range(random.randint(2, 3)):
                                    aug = apply_medical_augmentation(img_gray)
                                    aug.save(
                                        os.path.join(dst, f"{name}_aug{i + 1}{ext}")
                                    )
                            else:
                                for i in range(random.randint(4, 6)):
                                    aug = apply_medical_augmentation(img_gray)
                                    aug.save(
                                        os.path.join(dst, f"{name}_aug{i + 1}{ext}")
                                    )
                    except Exception as e:
                        logger.error(f"Augmentation failed for {f}: {e}")

    # Copy entire test set to validation
    copied = copy_files_between_splits(target_path, "test", "validation", classes)
    logger.info(f"Copied {copied} files from test to validation")

    return True


def count_files_in_dataset(path):
    base = os.path.join(path, "train")
    if not os.path.exists(base):
        print(f"Error: 'train' not found in {path}")
        return
    classes = os.listdir(base)
    for split in ["train", "validation", "test"]:
        total = 0
        print(f"\n{split.upper()}:")
        for c in classes:
            d = os.path.join(path, split, c)
            cnt = (
                len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
                if os.path.exists(d)
                else 0
            )
            print(f"  {c}: {cnt}")
            total += cnt
        print(f"  TOTAL: {total}")


def main():
    parser = argparse.ArgumentParser(
        "Chest X-ray dataset setup (optional augmentation)"
    )
    parser.add_argument(
        "--augment",
        type=int,
        choices=[0, 1],
        required=True,
        help="0 = no augmentation, 1 = apply augmentation",
    )
    args = parser.parse_args()

    original_dataset = "E:\medical_image_classification\labeled-chest-xray-images"
    target_dataset = "E:\medical_image_classification\splited_chest_xray_2"

    if os.path.exists(target_dataset):
        shutil.rmtree(target_dataset)
        logger.info(f"Removed existing: {target_dataset}")

    success = setup_dataset(original_dataset, target_dataset, args.augment)
    if success:
        print("\nDataset setup complete. File counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("Dataset setup failed. See logs.")


if __name__ == "__main__":
    main()
