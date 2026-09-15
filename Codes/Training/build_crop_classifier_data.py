"""
build_crop_classifier_data.py

Extracts padded crops around each `head`/`helmet` box from the 2-class
dataset and saves them into a folder-per-class image classification layout:

    dst/
      train/
        violation/   (from `head` boxes)
        compliant/    (from `helmet` boxes)
      valid/
        violation/
        compliant/
      test/
        violation/
        compliant/

This is Arm 2's stage-2 input: given a crop (produced by stage-1's
head-region detector at inference time, or ground-truth boxes here for
training), classify compliant vs. violation.

Padding: crops are expanded by `pad_frac` of the box's own size on each
side, clipped to image bounds, so the classifier sees some context
(shoulders, background) rather than a razor-tight headgear-only box —
closer to what stage-1's detector output will actually look like at
inference time.

Usage:
    python build_crop_classifier_data.py \
        --src Hard-Hat-Workers-10-2class \
        --dst crop_classifier_data \
        --pad_frac 0.25
"""

import argparse
import json
import os

import cv2


CLASS_MAP = {"head": "violation", "helmet": "compliant"}


def extract_split(src_dir, dst_dir, split, pad_frac):
    with open(os.path.join(src_dir, split, "_annotations.coco.json")) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_lookup = {img["id"]: img for img in coco["images"]}

    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    counts = {cls: 0 for cls in CLASS_MAP.values()}
    skipped_degenerate = 0

    # cache loaded images per image_id to avoid re-reading repeatedly
    img_cache = {}

    for ann in coco["annotations"]:
        class_name = cat_by_id[ann["category_id"]]
        if class_name not in CLASS_MAP:
            continue
        out_class = CLASS_MAP[class_name]

        img_info = img_lookup[ann["image_id"]]
        if img_info["id"] not in img_cache:
            img_cache[img_info["id"]] = cv2.imread(
                os.path.join(src_dir, split, img_info["file_name"])
            )
        img = img_cache[img_info["id"]]
        img_h, img_w = img.shape[:2]

        x, y, w, h = ann["bbox"]
        pad_x, pad_y = w * pad_frac, h * pad_frac
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))

        if x2 <= x1 or y2 <= y1:
            skipped_degenerate += 1
            continue

        crop = img[y1:y2, x1:x2]
        out_name = f"{img_info['id']}_{ann['id']}.jpg"
        out_path = os.path.join(dst_dir, split, out_class, out_name)
        cv2.imwrite(out_path, crop)
        counts[out_class] += 1

    print(f"[{split}] violation crops: {counts['violation']}, "
          f"compliant crops: {counts['compliant']}, "
          f"skipped degenerate: {skipped_degenerate}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--pad_frac", type=float, default=0.25,
                         help="Padding as a fraction of box width/height on each side")
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        extract_split(args.src, args.dst, split, args.pad_frac)


if __name__ == "__main__":
    main()
