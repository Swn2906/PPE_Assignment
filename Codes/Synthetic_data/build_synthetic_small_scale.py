"""
build_synthetic_small_scale.py

Constructs a resolution-degraded test slice from the natural test set, by
shrinking each image (and its boxes) by a shared factor, then letting the
model's own preprocessing upsample it back up at inference time. This
reduces the amount of real visual detail captured while preserving the
same relative composition/framing.

Images whose largest headgear box is already very small in absolute
pixels are left unchanged (nothing further to shrink toward).

Usage:
    python build_synthetic_small_scale.py \
        --src Hard-Hat-Workers-10-2class/test \
        --dst Hard-Hat-Workers-10-2class/test_resolution_degraded \
        --target_rel_h 0.021
"""

import argparse
import json
import os

import cv2
import numpy as np


def shrink_whole_frame(img, anns, target_rel_h):
    """Shrink the ENTIRE image+boxes by a shared factor, so the largest
    box's ABSOLUTE pixel height is reduced. No padding/canvas -- the output
    image is genuinely smaller in pixel dimensions, preserving full scene
    composition. Relative box height (box_h / img_h) is UNCHANGED by this
    transform (mathematically invariant to a shared scale factor) -- this
    method reduces absolute visual detail, not relative object-to-frame
    scale. See module docstring.

    Returns (new_img, new_anns, applied) where applied=False if no shrink was needed.
    """
    h, w = img.shape[:2]
    if not anns:
        return img, anns, False

    max_box_h = max(a["bbox"][3] for a in anns)
    cur_rel_h = max_box_h / h

    if cur_rel_h <= target_rel_h:
        return img, anns, False  # already small enough, leave as-is

    factor = target_rel_h / cur_rel_h
    new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
    small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    new_anns = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        new_ann = dict(a)
        new_ann["bbox"] = [x * factor, y * factor, bw * factor, bh * factor]
        new_anns.append(new_ann)

    return small_img, new_anns, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--target_rel_h", type=float, default=0.021,
                         help="Target box-height shrink factor basis (does NOT set final "
                              "relative scale -- see module docstring)")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    with open(os.path.join(args.src, "_annotations.coco.json")) as f:
        coco = json.load(f)

    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    new_images, new_annotations = [], []
    next_ann_id = 1
    n_shrunk = 0

    for img_info in coco["images"]:
        img_path = os.path.join(args.src, img_info["file_name"])
        img = cv2.imread(img_path)
        anns = by_image.get(img_info["id"], [])

        new_img, new_anns, applied = shrink_whole_frame(img, anns, args.target_rel_h)
        if applied:
            n_shrunk += 1

        out_path = os.path.join(args.dst, img_info["file_name"])
        cv2.imwrite(out_path, new_img)

        # record the ACTUAL new dimensions, since the frame genuinely shrank
        new_img_info = dict(img_info)
        new_img_info["height"], new_img_info["width"] = new_img.shape[0], new_img.shape[1]
        new_images.append(new_img_info)
        for a in new_anns:
            a2 = dict(a)
            a2["id"] = next_ann_id
            next_ann_id += 1
            new_annotations.append(a2)

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }
    with open(os.path.join(args.dst, "_annotations.coco.json"), "w") as f:
        json.dump(new_coco, f)

    print(f"Total images: {len(new_images)}")
    print(f"Images resolution-degraded: {n_shrunk} ({n_shrunk/len(new_images):.1%})")
    print(f"Images already small enough (left unchanged): {len(new_images) - n_shrunk}")
    print(f"Total annotations: {len(new_annotations)}")
    print("\nNote: this reduces absolute image detail/resolution; it does NOT change")
    print("relative box-to-frame scale (verified -- see EVAL_PLAN.md). Report results")
    print("from this slice as a resolution-degradation test, not a small-object test.")


if __name__ == "__main__":
    main()