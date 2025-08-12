#!/usr/bin/env python3
"""
Improved chest X-ray dataset setup + augmentation script with progress bars (tqdm).

Key improvements:
- deterministic seed control (--seed)
- robust centered scaling (no top-left crop bug)
- no repeated transforms per call (sample without replacement)
- safe directory handling (pathlib)
- proper Image.open context usage
- augmentation metadata (json sidecar) saved with each augmented image
- conservative elastic deformation
- CLAHE via OpenCV (if available)
- tqdm progress bars for copying/augmentation (graceful fallback if tqdm not installed)
- retains the "copy test -> validation" behavior (intentionally unchanged)

Usage:
    python prepare_dataset.py --augment 1 --force --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Optional imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

# tqdm import with safe fallback
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

    def _noop_tqdm(iterable=None, **kwargs):
        # simple passthrough generator
        return iterable

    tqdm = _noop_tqdm

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AugParams:
    """Record of chosen augmentation parameters for reproducibility."""

    techs: List[str]
    rotate_angle: Optional[float] = None
    scale_factor: Optional[float] = None
    contrast_factor: Optional[float] = None
    noise_type: Optional[str] = None
    noise_params: Optional[dict] = None
    clahe_params: Optional[dict] = None
    elastic_params: Optional[dict] = None


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def apply_medical_augmentation(image: Image.Image) -> Tuple[Image.Image, AugParams]:
    """
    Apply 2 to 4 distinct augmentation techniques on the input PIL image.
    Techniques: rotate, scale, contrast, noise (Poisson), CLAHE, elastic.

    Returns augmented image and AugParams describing what was done.
    """
    techniques = ["rotate", "scale", "contrast", "noise", "clahe", "elastic"]
    # Pick k distinct techniques (no repeats) where k in [2,4]
    k = random.randint(2, 4)
    selected = random.sample(techniques, k)

    augmented = image.copy().convert("L")  # Work in grayscale for CXR
    params = AugParams(techs=selected)

    for tech in selected:
        if tech == "rotate":
            angle = float(random.uniform(-10.0, 10.0))
            params.rotate_angle = angle
            rotated = augmented.rotate(angle, resample=Image.BICUBIC, expand=True)
            # center-fit back to original size
            augmented = ImageOps.fit(
                rotated, image.size, method=Image.BICUBIC, centering=(0.5, 0.5)
            )

        elif tech == "scale":
            scale_factor = float(random.uniform(0.9, 1.1))
            params.scale_factor = scale_factor
            w, h = augmented.size
            new_w, new_h = int(round(w * scale_factor)), int(round(h * scale_factor))
            scaled = augmented.resize((new_w, new_h), resample=Image.BICUBIC)

            # center-crop if larger, center-pad if smaller
            if new_w >= w or new_h >= h:
                # center-crop scaled image
                left = max(0, (new_w - w) // 2)
                top = max(0, (new_h - h) // 2)
                augmented = scaled.crop((left, top, left + w, top + h))
            else:
                # paste centered onto black canvas
                canvas = Image.new(augmented.mode, (w, h), 0)
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                canvas.paste(scaled, (left, top))
                augmented = canvas

        elif tech == "contrast":
            factor = float(random.uniform(0.8, 1.2))
            params.contrast_factor = factor
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(factor)

        elif tech == "noise":
            # Poisson noise applied to intensity (kept conservative)
            params.noise_type = "poisson"
            arr = np.array(augmented).astype(np.float32) / 255.0
            lam = np.clip(arr * 255.0, 0, 255.0)
            noisy = np.random.poisson(lam) / 255.0
            noisy_img = (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
            params.noise_params = {"method": "poisson", "lambda_scale": "255"}
            augmented = Image.fromarray(noisy_img)

        elif tech == "clahe":
            # CLAHE via OpenCV (if available) otherwise fallback to PIL equalize
            clip_limit = 2.0
            tile_grid = (8, 8)
            params.clahe_params = {"clipLimit": clip_limit, "tileGridSize": tile_grid}

            arr = np.array(augmented)
            if cv2 is None:
                logger.warning(
                    "cv2 not available; falling back to ImageOps.equalize for CLAHE step."
                )
                augmented = ImageOps.equalize(augmented)
            else:
                # ensure uint8
                if arr.dtype != np.uint8:
                    arr = (
                        (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                        if arr.max() <= 1.0
                        else arr.astype(np.uint8)
                    )
                cl = cv2.createCLAHE(
                    clipLimit=clip_limit, tileGridSize=tile_grid
                ).apply(arr)
                augmented = Image.fromarray(cl)

        elif tech == "elastic":
            # Conservative elastic deformation (only if scipy.ndimage present)
            if ndi is None:
                logger.warning(
                    "scipy.ndimage not available; skipping elastic augmentation."
                )
                params.elastic_params = {"applied": False}
                continue

            alpha = float(random.uniform(6.0, 12.0))  # strength
            sigma = float(random.uniform(3.0, 6.0))  # smoothness
            params.elastic_params = {"alpha": alpha, "sigma": sigma, "applied": True}

            arr = np.array(augmented).astype(np.float32)
            h, w = arr.shape[:2]

            # create displacement fields
            dx = np.random.rand(h, w) * 2 - 1
            dy = np.random.rand(h, w) * 2 - 1
            dx = ndi.gaussian_filter(dx, sigma=sigma, mode="reflect") * alpha
            dy = ndi.gaussian_filter(dy, sigma=sigma, mode="reflect") * alpha

            # meshgrid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.vstack(((y + dy).reshape(1, -1), (x + dx).reshape(1, -1)))

            # apply mapping (single channel)
            warped = ndi.map_coordinates(arr, indices, order=1, mode="reflect").reshape(
                h, w
            )
            warped_clipped = np.clip(warped, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(warped_clipped)

        else:
            raise RuntimeError(f"Unknown augmentation: {tech}")

    return augmented, params


def create_directory_structure(destination_root: Path, classes: List[str]):
    for split in ["train", "validation", "test"]:
        for class_name in classes:
            path = destination_root / split / class_name
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {path}")


def copy_files_between_splits(
    source_root: Path,
    split_from: str,
    split_to: str,
    classes: List[str],
    random_seed: int = 42,
) -> int:
    random.seed(random_seed)
    total = 0
    for class_name in classes:
        src_dir = source_root / split_from / class_name
        tgt_dir = source_root / split_to / class_name
        tgt_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            logger.warning(f"Source dir for copying does not exist: {src_dir}")
            continue
        files = [p.name for p in src_dir.iterdir() if p.is_file()]
        iter_files = (
            tqdm(files, desc=f"Copy {split_from}->{split_to} {class_name}", unit="file")
            if getattr(tqdm, "__name__", "") != "_noop_tqdm"
            else files
        )
        for fname in iter_files:
            try:
                shutil.copy2(src_dir / fname, tgt_dir / fname)
                total += 1
            except Exception as e:
                logger.error(f"Failed to copy {fname}: {e}")
    return total


def setup_dataset(
    original_path: Path,
    target_path: Path,
    augment_flag: int,
    seed: Optional[int] = None,
) -> bool:
    _set_seed(seed)

    train_dir = original_path / "train"
    if not train_dir.exists():
        logger.error(f"Train directory not found: {train_dir}")
        return False

    # classes = only directories inside original train
    classes = [p.name for p in train_dir.iterdir() if p.is_dir()]
    if not classes:
        logger.error(f"No class directories found under {train_dir}")
        return False

    logger.info(f"Found classes: {classes}")
    create_directory_structure(target_path, classes)

    for split in ["train", "test"]:
        for class_name in classes:
            src = original_path / split / class_name
            dst = target_path / split / class_name
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                logger.warning(f"Skipping missing source directory: {src}")
                continue

            files = [p for p in src.iterdir() if p.is_file()]
            desc = (
                f"Copy & Augment [{split}/{class_name}]"
                if split == "train" and augment_flag == 1
                else f"Copy [{split}/{class_name}]"
            )
            iter_files = (
                tqdm(files, desc=desc, unit="file")
                if getattr(tqdm, "__name__", "") != "_noop_tqdm"
                else files
            )

            logger.info(
                f"Copying {'and augmenting ' if split == 'train' and augment_flag == 1 else ''}{len(files)} files from {src} to {dst}"
            )

            for p in iter_files:
                try:
                    shutil.copy2(p, dst / p.name)
                except Exception as e:
                    logger.error(f"Failed to copy {p}: {e}")
                    continue

                if split == "train" and augment_flag == 1:
                    # Decide augmentation multiplicity based on class
                    if class_name.upper() == "PNEUMONIA":
                        aug_count = random.randint(2, 3)
                    else:
                        aug_count = random.randint(4, 6)

                    # open image safely
                    try:
                        with Image.open(p) as im:
                            img_gray = im.convert("L")
                            # augmentation inner loop progress (small bars)
                            aug_iter = (
                                tqdm(
                                    range(aug_count),
                                    desc=f"Aug {p.name}",
                                    leave=False,
                                    unit="aug",
                                )
                                if getattr(tqdm, "__name__", "") != "_noop_tqdm"
                                else range(aug_count)
                            )
                            for i in aug_iter:
                                try:
                                    aug_img, aug_params = apply_medical_augmentation(
                                        img_gray
                                    )
                                    name = p.stem
                                    ext = p.suffix or ".png"
                                    out_name = f"{name}_aug{i + 1}{ext}"
                                    out_path = dst / out_name
                                    aug_img.save(out_path)

                                    # Save augmentation metadata as json sidecar
                                    meta_path = dst / f"{name}_aug{i + 1}.json"
                                    with open(meta_path, "w") as jf:
                                        json.dump(asdict(aug_params), jf, indent=2)
                                except Exception as e:
                                    logger.error(
                                        f"Augmentation iteration failed for {p.name}: {e}"
                                    )
                    except Exception as e:
                        logger.error(f"Failed opening image {p}: {e}")

    # NOTE: intentionally preserving the original behavior: copy test -> validation
    copied = copy_files_between_splits(target_path, "test", "validation", classes)
    logger.info(f"Copied {copied} files from test to validation")

    return True


def count_files_in_dataset(path: Path):
    base = path / "train"
    if not base.exists():
        logger.error(f"Error: 'train' not found in {path}")
        return
    classes = [p.name for p in base.iterdir() if p.is_dir()]
    for split in ["train", "validation", "test"]:
        total = 0
        print(f"\n{split.upper()}:")
        for c in classes:
            d = path / split / c
            cnt = len([f for f in d.iterdir() if f.is_file()]) if d.exists() else 0
            print(f"  {c}: {cnt}")
            total += cnt
        print(f"  TOTAL: {total}")


def parse_args():
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If provided, will remove existing target dataset directory",
    )
    parser.add_argument(
        "--original",
        type=str,
        default=r"E:\medical_image_classification\labeled-chest-xray-images",
        help="Original dataset root",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=r"E:\medical_image_classification\splited_chest_xray_2",
        help="Target dataset root",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    original_dataset = Path(args.original)
    target_dataset = Path(args.target)

    if target_dataset.exists():
        if args.force:
            shutil.rmtree(target_dataset)
            logger.info(f"Removed existing: {target_dataset}")
        else:
            logger.error(
                f"Target dataset already exists: {target_dataset}. Use --force to remove."
            )
            return

    success = setup_dataset(
        original_dataset, target_dataset, args.augment, seed=args.seed
    )
    if success:
        print("\nDataset setup complete. File counts:")
        count_files_in_dataset(target_dataset)
    else:
        print("Dataset setup failed. See logs.")


if __name__ == "__main__":
    main()
