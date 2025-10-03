#!/usr/bin/env python3
"""
Rename images by prefixing with their parent folder name.

Example:
  train/3424234/1.jpg  ->  train/3424234/3424234_1.jpg

Usage:
  python prefix_with_class.py --root /path/to/dataset
  python prefix_with_class.py --root . --subsets train val --dry-run
"""

import argparse
from pathlib import Path

# Allowed file extensions (case-insensitive). Edit if you want other types.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def rename_in_subset(root: Path, subset: str, dry_run: bool = False, verbose: bool = True):
    subset_path = root / subset
    if not subset_path.exists() or not subset_path.is_dir():
        if verbose:
            print(f"Skipping missing subset: {subset_path}")
        return

    for class_dir in sorted(subset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if verbose:
            print(f"\nProcessing class folder: {class_dir}")

        for f in sorted(class_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in IMAGE_EXTS:
                # skip non-image files; remove this check if you want to rename all files
                continue

            orig_name = f.name

            # if already prefixed with the class name, skip
            if orig_name.startswith(f"{class_name}_"):
                if verbose:
                    print(f"  Skipping (already prefixed): {orig_name}")
                continue

            # propose new name: class_origname (preserve original extension)
            new_name = f"{class_name}_{orig_name}"
            dest = f.with_name(new_name)

            # If destination exists (unlikely if class names are unique), append counter
            if dest.exists():
                base = f.stem  # original name without suffix
                suffix = f.suffix
                counter = 1
                while True:
                    candidate_name = f"{class_name}_{base}_{counter}{suffix}"
                    candidate = f.with_name(candidate_name)
                    if not candidate.exists():
                        dest = candidate
                        break
                    counter += 1

            if dry_run:
                print(f"  Would rename: {orig_name} -> {dest.name}")
            else:
                f.rename(dest)
                if verbose:
                    print(f"  Renamed: {orig_name} -> {dest.name}")

def main():
    parser = argparse.ArgumentParser(description="Prefix image filenames with their parent folder name.")
    parser.add_argument("--root", type=str, default=".", help="Root folder containing 'train' and 'validation' (default: current dir).")
    parser.add_argument("--subsets", nargs="+", default=["train", "validation", "test"], help="Subfolders to process (default: train validation).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without renaming.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Less output.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.verbose:
        print(f"Root: {root}")
        print(f"Subsets: {args.subsets}")
        print(f"Dry run: {args.dry_run}")

    for subset in args.subsets:
        rename_in_subset(root, subset, dry_run=args.dry_run, verbose=args.verbose)

if __name__ == "__main__":
    main()
