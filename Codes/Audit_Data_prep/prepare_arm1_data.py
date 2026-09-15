"""
prepare_arm1_data.py

Converts the raw 3-class (head, helmet, person) COCO export into a 2-class
(head, helmet) dataset for Arm 1 (direct detection). Images with only
`person` annotations become zero-annotation (negative) images rather than
being dropped, since they're still valid images for the detector to see.

Usage:
    python prepare_arm1_data.py --src Hard-Hat-Workers-10 --dst Hard-Hat-Workers-10-2class
"""

import argparse
import json
import os
import shutil


DROP_CLASSES = {"person", "Workers"}  # "Workers" is Roboflow's unused supercategory placeholder (0 annotations)


def filter_split(src_dir, dst_dir, split):
    os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

    with open(os.path.join(src_dir, split, "_annotations.coco.json")) as f:
        coco = json.load(f)

    # Keep only categories not in DROP_CLASSES, remap to contiguous ids starting at 1
    # (COCO convention: category id 0 is reserved / background in many training libs)
    kept_categories = [c for c in coco["categories"] if c["name"] not in DROP_CLASSES]
    old_to_new_id = {}
    new_categories = []
    for new_id, cat in enumerate(kept_categories, start=1):
        old_to_new_id[cat["id"]] = new_id
        new_categories.append({
            "id": new_id,
            "name": cat["name"],
            "supercategory": cat.get("supercategory", "none"),
        })

    # Filter annotations: drop any belonging to a dropped class, remap the rest
    new_annotations = []
    next_ann_id = 1
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new_id:
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = old_to_new_id[ann["category_id"]]
        new_ann["id"] = next_ann_id
        next_ann_id += 1
        new_annotations.append(new_ann)

    new_coco = {
        "images": coco["images"],  # keep all images, including now-empty ones
        "annotations": new_annotations,
        "categories": new_categories,
    }

    with open(os.path.join(dst_dir, split, "_annotations.coco.json"), "w") as f:
        json.dump(new_coco, f)

    # Copy images
    for img in coco["images"]:
        src_path = os.path.join(src_dir, split, img["file_name"])
        dst_path = os.path.join(dst_dir, split, img["file_name"])
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

    n_images_now_empty = sum(
        1 for img in coco["images"]
        if not any(a["image_id"] == img["id"] for a in new_annotations)
    )

    print(f"[{split}] images: {len(coco['images'])}, "
          f"annotations kept: {len(new_annotations)}/{len(coco['annotations'])}, "
          f"now-empty images (had only person boxes): {n_images_now_empty}")

    return new_categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        cats = filter_split(args.src, args.dst, split)

    print("\nFinal category mapping (consistent across splits):")
    for c in cats:
        print(f"  {c['id']}: {c['name']}")


if __name__ == "__main__":
    main()