#!/usr/bin/env python3
"""
split_5_1.py

Usage examples:
  python split_5_1.py --src /path/to/dataset --dst /path/to/output
  python split_5_1.py --src dataset --dst out --strategy random --seed 42 --copy

Behavior:
 - Expects structure: src/<class_id>/*.jpg (or other image extensions)
 - For each class: put 5 images into dst/train/<class_id> and 1 image into dst/val/<class_id>.
 - Strategy "deterministic": uses sorted filenames (default).
 - Strategy "random": shuffles with provided seed (use --seed to make reproducible).
 - If a class has fewer than 6 images, script tries to put 1 image in val and rest in train (with warnings).
"""

import argparse
import os
import shutil
import random
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image_file(p: Path):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_class_images(img_paths, strategy='deterministic', seed=None):
    """
    Return (train_list, val_list) where train_list length is 5 and val_list length is 1 when possible.
    If fewer than 6 images exist, it will put 1 in val if possible and the rest in train.
    """
    imgs = list(img_paths)
    if strategy == 'random':
        rnd = random.Random(seed)
        rnd.shuffle(imgs)
    else:
        imgs = sorted(imgs, key=lambda p: p.name)

    n = len(imgs)
    if n >= 6:
        train = imgs[:5]
        val = imgs[5:6]
    elif n == 5:
        # no enough for 6, but follow "mostly train" idea: 4 train + 1 val
        train = imgs[:4]
        val = imgs[4:5]
    elif n >= 2:
        # put 1 to val, rest to train
        train = imgs[:-1]
        val = imgs[-1:]
    elif n == 1:
        train = []
        val = imgs[:]  # single image goes to val (warn)
    else:
        train = []
        val = []
    return train, val

def copy_or_move(files, dest_dir: Path, move=False, overwrite=False):
    ensure_dir(dest_dir)
    for src in files:
        dst = dest_dir / src.name
        if dst.exists():
            if overwrite:
                if dst.is_file():
                    dst.unlink()
            else:
                print(f"  - skipping (exists): {dst}")
                continue
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    train_root = dst / 'train'
    val_root = dst / 'val'
    ensure_dir(train_root)
    ensure_dir(val_root)

    class_folders = [p for p in src.iterdir() if p.is_dir()]
    if not class_folders:
        raise SystemExit(f"No class subfolders found in {src}")

    summary = {}
    for cls in sorted(class_folders, key=lambda p: p.name):
        images = [p for p in cls.iterdir() if is_image_file(p)]
        train_imgs, val_imgs = split_class_images(images, strategy=args.strategy, seed=args.seed)

        cls_train_dir = train_root / cls.name
        cls_val_dir = val_root / cls.name

        copy_or_move(train_imgs, cls_train_dir, move=args.move, overwrite=args.overwrite)
        copy_or_move(val_imgs, cls_val_dir, move=args.move, overwrite=args.overwrite)

        summary[cls.name] = (len(train_imgs), len(val_imgs), len(images))
        # quick status print
        print(f"Class {cls.name}: total={len(images)} -> train={len(train_imgs)}, val={len(val_imgs)}")

    # final report
    total_train = sum(v[0] for v in summary.values())
    total_val = sum(v[1] for v in summary.values())
    total_imgs = sum(v[2] for v in summary.values())
    print("="*40)
    print(f"Finished. Classes={len(summary)}  Images total={total_imgs}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Output root: {dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train (5 imgs/class) and val (1 img/class).")
    parser.add_argument('--src', required=True, help='Source dataset root (contains class subfolders).')
    parser.add_argument('--dst', required=True, help='Destination root where train/ and val/ will be created.')
    parser.add_argument('--strategy', choices=('deterministic', 'random'), default='deterministic',
                        help='How to choose images (default: deterministic by filename).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (when strategy=random).')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files in destination if they exist.')
    args = parser.parse_args()
    main(args)
