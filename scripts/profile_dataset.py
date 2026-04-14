"""
Data Profiling & Quality Assurance Script
Team jn — Crop Disease Detection Project

This script profiles the Multi-Crop Disease Dataset after extraction.
Run from the repository root:
    python3 scripts/profile_dataset.py

Expects the dataset to be extracted to data/ as described in data/README.md.
Dataset structure uses train/valid/test splits with YOLO annotations (data.yaml).
"""

import json
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path("data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_class_names(data_yaml: Path) -> list[str]:
    """Parse class names from data.yaml without requiring PyYAML."""
    names = []
    if not data_yaml.exists():
        return names
    with open(data_yaml) as f:
        in_names = False
        for line in f:
            line = line.strip()
            if line.startswith("names:"):
                # Names are on the same line as a Python list literal
                raw = line[len("names:"):].strip()
                raw = raw.strip("[]")
                names = [n.strip().strip("'\"") for n in raw.split(",")]
                break
    return names


def get_files(directory: Path, extensions: set) -> list[Path]:
    return [f for f in directory.rglob("*") if f.suffix.lower() in extensions]


def profile_split(split_dir: Path, class_names: list[str]) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = get_files(images_dir, IMAGE_EXTENSIONS) if images_dir.exists() else []
    labels = get_files(labels_dir, {".txt"}) if labels_dir.exists() else []

    image_stems = {f.stem: f for f in images}
    label_stems = {f.stem: f for f in labels}

    images_missing_label = [stem for stem in image_stems if stem not in label_stems]
    labels_missing_image = [stem for stem in label_stems if stem not in image_stems]

    empty_images = [str(f) for f in images if f.stat().st_size == 0]

    class_dist: dict[int, int] = defaultdict(int)
    annotation_count = 0
    for label_file in labels:
        try:
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_dist[class_id] += 1
                        annotation_count += 1
        except Exception as e:
            print(f"  [WARNING] Could not read {label_file}: {e}")

    return {
        "split": split_dir.name,
        "total_images": len(images),
        "total_labels": len(labels),
        "total_annotations": annotation_count,
        "images_missing_label_count": len(images_missing_label),
        "labels_missing_image_count": len(labels_missing_image),
        "empty_images": empty_images,
        "class_distribution": dict(class_dist),
    }


def run_profiling():
    print("=" * 60)
    print("Multi-Crop Disease Dataset — Profiling Report")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found: {DATA_DIR}")
        print("Please extract the dataset first. See data/README.md for instructions.")
        return

    class_names = load_class_names(DATA_DIR / "data.yaml")
    if class_names:
        print(f"\nClasses ({len(class_names)} total):")
        for i, name in enumerate(class_names):
            print(f"  {i:2d}: {name}")

    summary = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": [],
        "qa_issues": [],
    }

    total_images = 0
    total_annotations = 0
    combined_class_dist: dict[int, int] = defaultdict(int)

    for split_name in SPLITS:
        split_dir = DATA_DIR / split_name
        print(f"\n--- {split_name} split ---")

        if not split_dir.exists():
            print(f"  [WARNING] Split directory not found: {split_dir}")
            continue

        profile = profile_split(split_dir, class_names)
        summary["splits"].append(profile)
        total_images += profile["total_images"]
        total_annotations += profile["total_annotations"]

        for cls_id, count in profile["class_distribution"].items():
            combined_class_dist[cls_id] += count

        print(f"  Images:      {profile['total_images']}")
        print(f"  Labels:      {profile['total_labels']}")
        print(f"  Annotations: {profile['total_annotations']}")

        if profile["images_missing_label_count"] > 0:
            issue = f"{split_name}: {profile['images_missing_label_count']} image(s) missing a label file"
            summary["qa_issues"].append(issue)
            print(f"  [QA] {issue}")

        if profile["labels_missing_image_count"] > 0:
            issue = f"{split_name}: {profile['labels_missing_image_count']} label(s) missing an image file"
            summary["qa_issues"].append(issue)
            print(f"  [QA] {issue}")

        if profile["empty_images"]:
            issue = f"{split_name}: {len(profile['empty_images'])} empty/zero-byte image(s)"
            summary["qa_issues"].append(issue)
            print(f"  [QA] {issue}")

    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print(f"Total images:      {total_images}")
    print(f"Total annotations: {total_annotations}")

    print(f"\nAnnotations per class (all splits combined):")
    for cls_id in sorted(combined_class_dist):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        count = combined_class_dist[cls_id]
        bar = "#" * (count // 100)
        print(f"  {name:<45} {count:>6}  {bar}")

    if summary["qa_issues"]:
        print(f"\nQA Issues Found ({len(summary['qa_issues'])}):")
        for issue in summary["qa_issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo QA issues detected.")

    summary["total_images"] = total_images
    summary["total_annotations"] = total_annotations
    summary["combined_class_distribution"] = dict(combined_class_dist)

    output_path = Path("data/profile_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_profiling()
