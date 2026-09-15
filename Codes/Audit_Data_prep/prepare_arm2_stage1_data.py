"""
prepare_arm2_stage1_data.py

Merges `head` and `helmet` into a single class ("headgear") for Arm 2's
stage-1 detector. Stage 1 only needs to localize head-regions; the
compliant/violation decision is deferred to stage 2's crop classifier.

Reuses the same 2-class dataset (Hard-Hat-Workers-10-2class) as the source,
so both arms train on an identical image set and label cleaning decision,
per the assignment's comparability requirement.

Usage:
    python prepare_arm2_stage1_data.py \
        --src Hard-Hat-Workers-10-2class \
        --dst Hard-Hat-Workers-10-headgear
"""

import argparse
import json
import os
import shutil


def merge_split(src_dir, dst_dir, split):
    os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

    with open(os.path.join(src_dir, split, "_annotations.coco.json")) as f:
        coco = json.load(f)

    # Single merged category
    new_categories = [{"id": 1, "name": "headgear", "supercategory": "none"}]

    new_annotations = []
    for next_id, ann in enumerate(coco["annotations"], start=1):
        new_ann = dict(ann)
        new_ann["category_id"] = 1  # merge head + helmet -> single class
        new_ann["id"] = next_id
        new_annotations.append(new_ann)

    new_coco = {
        "images": coco["images"],
        "annotations": new_annotations,
        "categories": new_categories,
    }

    with open(os.path.join(dst_dir, split, "_annotations.coco.json"), "w") as f:
        json.dump(new_coco, f)

    for img in coco["images"]:
        src_path = os.path.join(src_dir, split, img["file_name"])
        dst_path = os.path.join(dst_dir, split, img["file_name"])
        if not os.path.exists(dst_path):
            # symlink instead of copy: images are identical to the source
            # dataset, only annotations differ. Copying thousands of files
            # onto a Drive mount is slow; symlinks are near-instant.
            os.symlink(os.path.abspath(src_path), dst_path)

    print(f"[{split}] images: {len(coco['images'])}, annotations (merged): {len(new_annotations)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        merge_split(args.src, args.dst, split)

    print("\nMerged category: 1: headgear")


if __name__ == "__main__":
    main()
