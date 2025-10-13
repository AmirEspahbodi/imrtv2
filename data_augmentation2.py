"""
Augmentation pipeline for product image retrieval (foreground-aware, conservative photometric changes).

- Preserves color/detail as much as possible (small photometric ranges)
- Uses foreground mask extracted from near-white background (since your images were composited to white)
- Applies geometric transforms to both image and mask, then composites onto backgrounds (if provided)
- Adds subtle shadows and tiny JPEG/noise artifacts to mimic user photos
- Generates N augmentations per original (default 25)

Usage:
    python augmentation_pipeline.py --input_dir dataset --output_dir augmented_dataset --n_per_image 25 --backgrounds backgrounds_dir

Requirements:
    pip install numpy opencv-python pillow albumentations tqdm

"""

import os
import uuid
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import albumentations as A


# ------------------------ Helper functions ------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_image_cv2(path: Path) -> np.ndarray:
    # Read as BGR (cv2 default) then convert to RGB
    img_bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def write_jpeg_cv2(path: Path, image_rgb: np.ndarray, quality: int = 95):
    # cv2.imencode expects BGR
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(str(path))[-1].lower()
    if ext not in ['.jpg', '.jpeg']:
        # force jpg extension
        path = Path(str(path) + '.jpg')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    with open(path, 'wb') as f:
        encimg.tofile(f)


def mask_from_white_bg(image_rgb: np.ndarray, white_thresh: int = 245, dilate_iter: int = 3, blur_ksize: int = 5) -> np.ndarray:
    """
    Create a foreground mask by thresholding near-white background.
    Returns mask uint8 0/255 shape HxW.
    """
    # image_rgb in 0..255
    assert image_rgb.dtype == np.uint8
    # Detect non-white pixels
    white_mask = np.logical_and.reduce((image_rgb[:, :, 0] >= white_thresh,
                                        image_rgb[:, :, 1] >= white_thresh,
                                        image_rgb[:, :, 2] >= white_thresh))
    fg_mask = (~white_mask).astype(np.uint8) * 255

    # Morphological cleanup
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=dilate_iter)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Slight blur to soften edges for nicer composites
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        fg_mask = cv2.GaussianBlur(fg_mask, (k, k), 0)

    return fg_mask


def add_shadow(background_rgb: np.ndarray, fg_mask: np.ndarray, offset: Tuple[int, int] = (8, 8), blur_ksize: int = 31, shadow_strength: float = 0.6) -> np.ndarray:
    """
    Paint a soft shadow onto the background using the fg_mask.
    offset: (dx, dy) in pixels.
    blur_ksize: should be large (odd) for soft shadow.
    shadow_strength: 0..1 how dark the shadow is.
    """
    h, w = fg_mask.shape
    # create shadow mask
    shadow = np.zeros_like(fg_mask, dtype=np.uint8)
    dx, dy = offset
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(fg_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    shadow = cv2.GaussianBlur(shifted, (k, k), 0)
    shadow_norm = (shadow.astype(np.float32) / 255.0) * shadow_strength

    # apply shadow by darkening background
    bg_float = background_rgb.astype(np.float32) / 255.0
    # ensure shadow acts multiplicatively on luminance; convert to gray mask
    shadow_gray = shadow_norm[..., None]
    bg_with_shadow = bg_float * (1 - shadow_gray * 0.6)  # small darkening
    bg_with_shadow = np.clip(bg_with_shadow * 255.0, 0, 255).astype(np.uint8)
    return bg_with_shadow


def composite_foreground(background_rgb: np.ndarray, fg_rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Composite fg onto background using soft alpha mask (0..255)
    Both images must be same size.
    """
    if fg_rgb.shape[:2] != background_rgb.shape[:2]:
        # resize background to fg size
        background_rgb = cv2.resize(background_rgb, (fg_rgb.shape[1], fg_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    alpha = (fg_mask.astype(np.float32) / 255.0)[..., None]
    comp = (fg_rgb.astype(np.float32) * alpha + background_rgb.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return comp


# ------------------------ Augmentation pipeline ------------------------


def build_albumentations_transform():
    """Construct the conservative augmentation pipeline reflecting the recommendations."""
    # geometric transforms applied to image + mask
    geom = [
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=12, p=0.8, border_mode=cv2.BORDER_CONSTANT),
        A.Perspective(scale=(0.01, 0.04), p=0.3),
        # Horizontal flip safe only sometimes; keep low prob
        A.HorizontalFlip(p=0.3),
    ]

    photometric = [
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=3, val_shift_limit=0, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.25),
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 3)),
            A.MedianBlur(blur_limit=3),
            A.ImageCompression(quality_lower=85, quality_upper=100),
        ], p=0.25),
    ]

    transform = A.Compose(geom + photometric, additional_targets={'mask': 'mask'})
    return transform


# ------------------------ Main generation function ------------------------

def generate_augmentations_for_image(
    src_path: Path,
    dst_dir: Path,
    transform: A.Compose,
    n_augs: int = 25,
    backgrounds: Optional[list] = None,
    keep_size: bool = True,
    quality: int = 95,
):
    """
    Generate n_augs augmented images for a single source image and save into dst_dir.
    """
    img = read_image_cv2(src_path)
    h, w = img.shape[:2]
    fg_mask = mask_from_white_bg(img, white_thresh=245, dilate_iter=2, blur_ksize=3)

    # prepare a pure white background fallback
    white_bg = np.ones_like(img, dtype=np.uint8) * 255

    for i in range(n_augs):
        # Sample background
        if backgrounds and np.random.rand() < 0.25:
            bg = backgrounds[np.random.randint(len(backgrounds))]
            # random crop/resize of background to the target size
            bh, bw = bg.shape[:2]
            if bh < h or bw < w:
                # upscale background slightly if too small
                bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # random crop
                y = np.random.randint(0, bh - h + 1) if bh > h else 0
                x = np.random.randint(0, bw - w + 1) if bw > w else 0
                bg_resized = bg[y:y+h, x:x+w]
            background = bg_resized
        else:
            background = white_bg

        # Random offset for shadow
        dx = int(np.round(np.random.uniform(4, 14))) * (1 if np.random.rand() < 0.7 else -1)
        dy = int(np.round(np.random.uniform(4, 20)))

        bg_with_shadow = add_shadow(background, fg_mask, offset=(dx, dy), blur_ksize=31, shadow_strength=np.random.uniform(0.35, 0.7))

        # Apply albumentations to image and mask (geometric/photometric)
        augmented = transform(image=img, mask=fg_mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        # Slightly jitter mask edges to avoid hard seams (small blur)
        mask_aug = cv2.GaussianBlur(mask_aug, (3, 3), 0)

        # Composite
        composite = composite_foreground(bg_with_shadow, img_aug, mask_aug)

        # Additional tiny JPEG artifacting (encode/decode) to simulate uploads
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(88, quality+1)]
        success, encimg = cv2.imencode('.jpg', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR), encode_param)
        if not success:
            raise RuntimeError('JPEG encode failed')
        decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        # Save with unique name to avoid collisions across classes
        uid = uuid.uuid4().hex[:8]
        dst_name = f"{src_path.stem}_aug_{i:03d}_{uid}.jpg"
        dst_path = dst_dir / dst_name
        write_jpeg_cv2(dst_path, decoded, quality=quality)


# ------------------------ Background loader ------------------------

def load_backgrounds_from_folder(folder: Path, min_size: Tuple[int, int] = (400, 400)) -> list:
    if not folder or not folder.exists():
        return []
    imgs = []
    for p in folder.rglob('*'):
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        img = read_image_cv2(p)
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            continue
        imgs.append(img)
    return imgs


# ------------------------ Directory walker & orchestrator ------------------------

def process_dataset(
    input_dir: Path,
    output_dir: Path,
    n_per_image: int = 25,
    backgrounds_dir: Optional[Path] = None,
):
    transform = build_albumentations_transform()
    backgrounds = load_backgrounds_from_folder(backgrounds_dir) if backgrounds_dir else []

    # Walk classes (top-level directories) and image files
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        rel_class = class_dir.name
        dst_class_dir = output_dir / rel_class
        ensure_dir(dst_class_dir)

        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        for img_path in tqdm(image_paths, desc=f"Class {rel_class}"):
            try:
                generate_augmentations_for_image(
                    src_path=img_path,
                    dst_dir=dst_class_dir,
                    transform=transform,
                    n_augs=n_per_image,
                    backgrounds=backgrounds,
                )
            except Exception as e:
                print(f"Failed augmenting {img_path}: {e}")


# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Conservative foreground-aware augmentation for product images')
    p.add_argument('--input_dir', type=Path, required=True, help='Path to input dataset root (folders per class)')
    p.add_argument('--output_dir', type=Path, required=True, help='Path to write augmented_dataset (keeps class folder names)')
    p.add_argument('--n_per_image', type=int, default=25, help='Number of augmented images to generate per source image (20-30 recommended)')
    p.add_argument('--backgrounds', type=Path, default=None, help='Optional folder with background images to composite onto')
    p.add_argument('--quality', type=int, default=95, help='JPEG quality for saved images')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    n = args.n_per_image
    bg_dir = args.backgrounds

    if not input_dir.exists():
        raise RuntimeError(f"Input dir does not exist: {input_dir}")
    ensure_dir(output_dir)

    backgrounds = []
    if bg_dir:
        backgrounds = load_backgrounds_from_folder(bg_dir)
        print(f"Loaded {len(backgrounds)} background images from {bg_dir}")

    process_dataset(input_dir, output_dir, n_per_image=n, backgrounds_dir=bg_dir)
    print('Done.')


if __name__ == '__main__':
    main()
