"""
Day 1: build data/processed/{train,val,test}/<class>/ from data/raw/<class>/.

Steps, in order:
  1. Collect images per class, verify each one decodes.
  2. Drop exact duplicates (identical file content) within each class.
  3. Split train/val/test with a fixed seed, stratified by class -
     BEFORE any augmentation exists, so no near-duplicate ever appears
     in two splits.
  4. Copy val/test images through unchanged - they must reflect real
     phone photos, since that's what the model will see in production.
  5. For train only, optionally add a background-removed copy of each
     image alongside the original, so the model learns both looks.

Run:
    python training/prepare_data.py
"""

import hashlib
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_config() -> dict:
    with open(REPO_ROOT / "training" / "config.yaml") as f:
        return yaml.safe_load(f)


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def collect_class_images(class_dir: Path) -> list[Path]:
    """List valid, de-duplicated image paths for one class."""
    candidates = [
        p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    ]

    valid = [p for p in candidates if is_valid_image(p)]
    skipped = len(candidates) - len(valid)
    if skipped:
        print(f"    skipped {skipped} unreadable/corrupt file(s)")

    seen_hashes: dict[str, Path] = {}
    unique: list[Path] = []
    for p in valid:
        h = file_hash(p)
        if h in seen_hashes:
            continue
        seen_hashes[h] = p
        unique.append(p)

    dupes = len(valid) - len(unique)
    if dupes:
        print(f"    removed {dupes} exact duplicate(s)")

    return unique


def split_paths(
    paths: list[Path], split: dict, seed: int
) -> dict[str, list[Path]]:
    """Shuffle deterministically and cut into train/val/test."""
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * split["train"])
    n_val = round(n * split["val"])
    # test gets the remainder, so rounding never drops an image
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def remove_background(src: Path, dst: Path) -> bool:
    """Save a background-removed copy of `src` to `dst`. Returns False
    (and leaves dst unwritten) if rembg isn't installed - background
    removal is an enhancement, not a hard requirement."""
    try:
        from rembg import remove
    except ImportError:
        return False

    with open(src, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)
    with open(dst, "wb") as f:
        f.write(output_bytes)
    return True


def main() -> None:
    config = load_config()
    random.seed(config["seed"])

    raw_dir = REPO_ROOT / config["data"]["raw_dir"]
    processed_dir = REPO_ROOT / config["data"]["processed_dir"]
    classes = config["data"]["classes"]
    min_images = config["data"]["min_images_per_class"]
    add_bg_removed = config["data"]["clean_background"]

    if processed_dir.exists():
        print(f"Clearing existing {processed_dir}")
        shutil.rmtree(processed_dir)

    bg_removal_available = None  # decided on first use, logged once
    counts: dict[str, dict[str, int]] = {}

    for class_name in classes:
        class_dir = raw_dir / class_name
        print(f"\n[{class_name}]")

        if not class_dir.is_dir():
            print(f"  WARNING: {class_dir} does not exist - skipping this class.")
            continue

        images = collect_class_images(class_dir)
        print(f"  {len(images)} usable image(s)")

        if len(images) < min_images:
            print(
                f"  WARNING: only {len(images)} images "
                f"(min_images_per_class is {min_images}). Results for this "
                f"class will likely be unreliable."
            )

        splits = split_paths(images, config["data"]["split"], config["seed"])
        counts[class_name] = {}

        for split_name, split_paths_list in splits.items():
            out_dir = processed_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for src in split_paths_list:
                shutil.copy2(src, out_dir / src.name)

            n_bg = 0
            if split_name == "train" and add_bg_removed:
                for src in split_paths_list:
                    dst = out_dir / f"{src.stem}_bgremoved.png"
                    ok = remove_background(src, dst)
                    if bg_removal_available is None:
                        bg_removal_available = ok
                        if not ok:
                            print(
                                "  NOTE: rembg not installed - skipping "
                                "background removal. `pip install rembg` "
                                "to enable it."
                            )
                    if ok:
                        n_bg += 1

            total = len(split_paths_list) + n_bg
            counts[class_name][split_name] = total
            extra = f" (+{n_bg} background-removed)" if n_bg else ""
            print(f"  {split_name}: {len(split_paths_list)}{extra} -> {total} total")

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    counts_path = REPO_ROOT / config["paths"]["class_counts_path"]
    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"\nDone. Class counts written to {counts_path}")
    print("Paste this table into README.md's Dataset section:\n")
    header = f"| Class | {' | '.join(c.capitalize() for c in ['train', 'val', 'test'])} |"
    print(header)
    print("|" + "---|" * (len(header.split('|')) - 2))
    for class_name, split_counts in counts.items():
        row = " | ".join(str(split_counts.get(s, 0)) for s in ["train", "val", "test"])
        print(f"| {class_name} | {row} |")


if __name__ == "__main__":
    main()
