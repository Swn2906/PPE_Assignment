"""
audit_utils.py

Reusable utilities for auditing the Hard Hat Workers v10 (COCO format) dataset.
Covers: loading, visualization, scale analysis, and structural consistency checks.

"""

import json
import os
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Consistent color scheme used across all visualizations in this project.
COLORS = {"head": (255, 0, 0), "helmet": (0, 255, 0), "person": (0, 0, 255)}
FLAG_COLOR = (255, 0, 255)  # magenta highlight for flagged boxes


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_coco(json_path):
    """Load a COCO annotation file and return convenience lookups.

    Returns:
        coco: raw parsed json dict
        by_image: dict[image_id] -> list of annotation dicts
        cat_by_id: dict[category_id] -> class name
        img_id_lookup: dict[image_id] -> image info dict
    """
    with open(json_path) as f:
        coco = json.load(f)

    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_lookup = {img["id"]: img for img in coco["images"]}

    return coco, by_image, cat_by_id, img_id_lookup


def class_counts(coco):
    """Return {class_name: count} for a loaded COCO dict."""
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(ann["category_id"] for ann in coco["annotations"])
    return {cat_by_id[k]: v for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def to_xyxy(bbox):
    """Convert COCO [x, y, w, h] bbox to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(a, b):
    """IoU between two COCO-format [x, y, w, h] boxes."""
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def contains_center(outer, inner):
    """True if the center of `inner` (COCO bbox) falls inside `outer` (COCO bbox)."""
    ox1, oy1, ox2, oy2 = to_xyxy(outer)
    ix1, iy1, ix2, iy2 = to_xyxy(inner)
    cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


# ---------------------------------------------------------------------------
# Scale analysis
# ---------------------------------------------------------------------------

def build_scale_dataframe(coco):
    """Build a per-annotation dataframe with absolute + relative box scale.

    Columns: class, box_h, box_w, img_h, rel_h (box_h / img_h)
    """
    img_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    records = []
    for ann in coco["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3]
        img_w, img_h = img_dims[ann["image_id"]]
        records.append({
            "class": cat_names[ann["category_id"]],
            "box_h": h,
            "box_w": w,
            "img_h": img_h,
            "rel_h": h / img_h,
        })
    return pd.DataFrame(records)


def add_scale_buckets(df, bins=(0, 15, 30, 60, 100, 10000),
                       labels=("<15px", "15-30px", "30-60px", "60-100px", ">100px")):
    """Add an absolute-pixel scale_bucket column to a scale dataframe (in place-safe)."""
    df = df.copy()
    df["scale_bucket"] = pd.cut(df["box_h"], bins=list(bins), labels=list(labels))
    return df


def deployment_scale_fraction(df, cls="head", rel_low=0.014, rel_high=0.028):
    """Fraction of boxes of `cls` whose relative height falls in the deployment band.

    Default band (1.4%-2.8%) corresponds to a 15-30px head in a 1080p (1080-tall) frame.
    """
    sub = df[df["class"] == cls]
    band = sub[sub["rel_h"].between(rel_low, rel_high)]
    return len(band) / len(sub) if len(sub) else float("nan")


# ---------------------------------------------------------------------------
# Structural consistency checks
# ---------------------------------------------------------------------------

def structural_consistency_report(coco, by_image, cat_by_id,
                                   headgear_classes=("head", "helmet"),
                                   dup_iou_thresh=0.9,
                                   degenerate_min_px=2,
                                   degenerate_aspect=8):
    """Run the four structural consistency checks and return a results dict
    plus the raw list of (image_id, annotation) tuples for cases worth inspecting.
    """
    persons_without_headgear = 0
    total_persons = 0
    headgear_without_person = 0
    total_headgear = 0
    duplicates = 0
    degenerate = 0

    persons_without_headgear_list = []

    for img_id, anns in by_image.items():
        persons = [a for a in anns if cat_by_id[a["category_id"]] == "person"]
        headgear = [a for a in anns if cat_by_id[a["category_id"]] in headgear_classes]

        total_persons += len(persons)
        for p in persons:
            if not any(contains_center(p["bbox"], hg["bbox"]) for hg in headgear):
                persons_without_headgear += 1
                persons_without_headgear_list.append((img_id, p))

        total_headgear += len(headgear)
        for hg in headgear:
            if not any(contains_center(p["bbox"], hg["bbox"]) for p in persons):
                headgear_without_person += 1

        # duplicates: same class, IoU > threshold
        by_cat = defaultdict(list)
        for a in anns:
            by_cat[a["category_id"]].append(a)
        for cls_anns in by_cat.values():
            for i in range(len(cls_anns)):
                for j in range(i + 1, len(cls_anns)):
                    if iou(cls_anns[i]["bbox"], cls_anns[j]["bbox"]) > dup_iou_thresh:
                        duplicates += 1

        # degenerate boxes
        for a in anns:
            w, h = a["bbox"][2], a["bbox"][3]
            if w < degenerate_min_px or h < degenerate_min_px:
                degenerate += 1
            elif w / h > degenerate_aspect or h / w > degenerate_aspect:
                degenerate += 1

    report = {
        "persons_without_headgear": persons_without_headgear,
        "total_persons": total_persons,
        "persons_without_headgear_pct": persons_without_headgear / total_persons if total_persons else float("nan"),
        "headgear_without_person": headgear_without_person,
        "total_headgear": total_headgear,
        "headgear_without_person_pct": headgear_without_person / total_headgear if total_headgear else float("nan"),
        "near_duplicates": duplicates,
        "degenerate_boxes": degenerate,
    }
    return report, persons_without_headgear_list


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_annotations(img_path, anns, cat_by_id, flagged_ann=None):
    """Return an RGB image (numpy array) with all boxes in `anns` drawn.

    If `flagged_ann` is given, it is additionally outlined in thick magenta.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        name = cat_by_id[ann["category_id"]]
        color = COLORS.get(name, (255, 255, 0))
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    if flagged_ann is not None:
        x, y, w, h = flagged_ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), FLAG_COLOR, 4)

    return img


def visualize_sample_grid(coco_images, by_image, cat_by_id, image_dir,
                           n=12, cols=4, seed=None, save_path=None,
                           flagged_lookup=None):
    """Draw a grid of randomly (or explicitly) sampled images with annotations.

    coco_images: list of image info dicts to sample from (e.g. coco["images"])
    flagged_lookup: optional dict[image_id] -> flagged annotation to highlight
    """
    import random
    if seed is not None:
        random.seed(seed)

    if len(coco_images) > n:
        sample_imgs = random.sample(coco_images, n)
    else:
        sample_imgs = coco_images

    rows = (len(sample_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flat if len(sample_imgs) > 1 else [axes]

    for ax, img_info in zip(axes, sample_imgs):
        img_path = os.path.join(image_dir, img_info["file_name"])
        anns = by_image.get(img_info["id"], [])
        flagged = flagged_lookup.get(img_info["id"]) if flagged_lookup else None
        img = draw_annotations(img_path, anns, cat_by_id, flagged_ann=flagged)
        ax.imshow(img)
        ax.set_title(img_info["file_name"], fontsize=8)
        ax.axis("off")

    # hide any unused subplots
    for ax in list(axes)[len(sample_imgs):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()


def dump_flagged_images(flagged_list, by_image, cat_by_id, img_id_lookup,
                         image_dir, out_dir):
    """Save individually annotated copies of every (image_id, flagged_ann) pair.

    flagged_list: list of (image_id, flagged_annotation) tuples,
                  e.g. output of structural_consistency_report(...)[1]
    """
    os.makedirs(out_dir, exist_ok=True)
    for img_id, flagged_ann in flagged_list:
        img_info = img_id_lookup[img_id]
        img_path = os.path.join(image_dir, img_info["file_name"])
        anns = by_image[img_id]
        img = draw_annotations(img_path, anns, cat_by_id, flagged_ann=flagged_ann)
        out_path = os.path.join(out_dir, img_info["file_name"])
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
