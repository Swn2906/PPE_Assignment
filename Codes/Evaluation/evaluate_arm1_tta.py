"""
evaluate_arm1_tta.py

Test-time augmentation experiment for Arm 1. Runs inference on three views
per image -- original, horizontally flipped, and 2x upscaled -- maps each
view's boxes back to the original image's coordinate frame, and merges
overlapping detections via confidence-weighted box averaging (a simplified
Weighted Boxes Fusion). This is a targeted experiment testing whether TTA
recovers any of the small-object recall collapse found in the non-TTA
evaluation, NOT a general accuracy-maximization pass.

Usage:
    python evaluate_arm1_tta.py \
        --checkpoint arm1_output/checkpoint_best_ema.pth \
        --dataset_dir Hard-Hat-Workers-10-2class/test_synthetic_small_without_padding \
        --conf_thresholds 0.3 0.5 \
        --upscale_factor 2.0 \
        --merge_iou 0.5
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np


SCALE_BINS = [0, 15, 30, 60, 100, 10000]
SCALE_LABELS = ["<15px", "15-30px", "30-60px", "60-100px", ">100px"]


def to_xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def scale_bucket(box_h):
    for lo, hi, label in zip(SCALE_BINS, SCALE_BINS[1:], SCALE_LABELS):
        if lo <= box_h < hi:
            return label
    return SCALE_LABELS[-1]


def get_multiview_detections(model, img, min_conf, upscale_factor):
    """Runs inference on 3 views of `img` (BGR numpy array) and returns all
    detections mapped back to the ORIGINAL image's coordinate frame, as a
    list of dicts: {bbox: xyxy, class: name, conf: float}.
    """
    h, w = img.shape[:2]
    all_dets = []

    # --- View 1: original ---
    dets = model.predict(img, threshold=min_conf)
    for i in range(len(dets.xyxy)):
        class_id = int(dets.class_id[i])
        name = model.class_names[class_id]
        all_dets.append({"bbox": tuple(dets.xyxy[i]), "class": name, "conf": float(dets.confidence[i])})

    # --- View 2: horizontal flip ---
    flipped = cv2.flip(img, 1)
    dets = model.predict(flipped, threshold=min_conf)
    for i in range(len(dets.xyxy)):
        x1, y1, x2, y2 = dets.xyxy[i]
        # map flipped-image coords back to original orientation
        orig_x1, orig_x2 = w - x2, w - x1
        class_id = int(dets.class_id[i])
        name = model.class_names[class_id]
        all_dets.append({"bbox": (orig_x1, y1, orig_x2, y2), "class": name, "conf": float(dets.confidence[i])})

    # --- View 3: upscaled ---
    new_w, new_h = int(w * upscale_factor), int(h * upscale_factor)
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    dets = model.predict(upscaled, threshold=min_conf)
    for i in range(len(dets.xyxy)):
        x1, y1, x2, y2 = dets.xyxy[i]
        # map back down to original scale
        orig_box = (x1 / upscale_factor, y1 / upscale_factor, x2 / upscale_factor, y2 / upscale_factor)
        class_id = int(dets.class_id[i])
        name = model.class_names[class_id]
        all_dets.append({"bbox": orig_box, "class": name, "conf": float(dets.confidence[i])})

    return all_dets


def merge_detections(dets, merge_iou):
    """Confidence-weighted box fusion: greedily group same-class boxes with
    IoU >= merge_iou, replace each group with one confidence-weighted-average
    box, score = max confidence in the group.
    """
    dets = sorted(dets, key=lambda d: -d["conf"])
    merged = []
    used = [False] * len(dets)

    for i, d in enumerate(dets):
        if used[i]:
            continue
        group = [d]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j] or dets[j]["class"] != d["class"]:
                continue
            if iou_xyxy(d["bbox"], dets[j]["bbox"]) >= merge_iou:
                group.append(dets[j])
                used[j] = True

        total_conf = sum(g["conf"] for g in group)
        avg_box = tuple(
            sum(g["bbox"][k] * g["conf"] for g in group) / total_conf
            for k in range(4)
        )
        merged.append({"bbox": avg_box, "class": d["class"], "conf": max(g["conf"] for g in group)})

    return merged


def run_evaluation(model, dataset_dir, min_conf_for_views, conf_thresholds, merge_iou, upscale_factor, iou_thresh=0.5):
    with open(os.path.join(dataset_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    # Run multi-view + merge ONCE per image (at a low min_conf to keep
    # candidate boxes available for the merge step), cache result, then
    # sweep final confidence thresholds cheaply.
    cached_merged = {}
    for img_info in coco["images"]:
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        raw_dets = get_multiview_detections(model, img, min_conf_for_views, upscale_factor)
        merged = merge_detections(raw_dets, merge_iou)
        cached_merged[img_info["id"]] = merged

    results = {}
    for conf_thresh in conf_thresholds:
        violation_bucket_stats = defaultdict(lambda: {"tp": 0, "fn": 0})
        head_tp, head_fp = 0, 0
        helmet_tp, helmet_fp, helmet_fn = 0, 0, 0

        for img_info in coco["images"]:
            anns = gt_by_image.get(img_info["id"], [])
            gts = []
            for a in anns:
                name = cat_by_id[a["category_id"]]
                gts.append({"bbox": to_xyxy(a["bbox"]), "class": name, "h": a["bbox"][3], "matched": False})

            preds = [d for d in cached_merged[img_info["id"]] if d["conf"] >= conf_thresh]
            preds.sort(key=lambda p: -p["conf"])

            for p in preds:
                best_iou, best_gt = 0, None
                for g in gts:
                    if g["matched"] or g["class"] != p["class"]:
                        continue
                    i = iou_xyxy(p["bbox"], g["bbox"])
                    if i > best_iou:
                        best_iou, best_gt = i, g
                is_tp = best_gt is not None and best_iou >= iou_thresh
                if is_tp:
                    best_gt["matched"] = True
                if p["class"] == "head":
                    head_tp += 1 if is_tp else 0
                    head_fp += 0 if is_tp else 1
                elif p["class"] == "helmet":
                    helmet_tp += 1 if is_tp else 0
                    helmet_fp += 0 if is_tp else 1

            for g in gts:
                if g["class"] == "head":
                    bucket = scale_bucket(g["h"])
                    if g["matched"]:
                        violation_bucket_stats[bucket]["tp"] += 1
                    else:
                        violation_bucket_stats[bucket]["fn"] += 1
                elif g["class"] == "helmet" and not g["matched"]:
                    helmet_fn += 1

        total_head_tp = sum(v["tp"] for v in violation_bucket_stats.values())
        total_head_fn = sum(v["fn"] for v in violation_bucket_stats.values())
        missed_violation_rate = (total_head_fn / (total_head_tp + total_head_fn)
                                  if (total_head_tp + total_head_fn) else float("nan"))
        false_alarm_rate = (head_fp / (head_tp + head_fp) if (head_tp + head_fp) else float("nan"))
        per_bucket_recall = {
            bucket: (v["tp"] / (v["tp"] + v["fn"]) if (v["tp"] + v["fn"]) else float("nan"))
            for bucket, v in violation_bucket_stats.items()
        }

        results[conf_thresh] = {
            "missed_violation_rate": missed_violation_rate,
            "false_alarm_rate": false_alarm_rate,
            "per_scale_bucket_recall": per_bucket_recall,
            "helmet_precision": helmet_tp / (helmet_tp + helmet_fp) if (helmet_tp + helmet_fp) else float("nan"),
            "helmet_recall": helmet_tp / (helmet_tp + helmet_fn) if (helmet_tp + helmet_fn) else float("nan"),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--conf_thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--min_conf_for_views", type=float, default=0.15,
                         help="Low threshold used when gathering per-view candidates, before merging")
    parser.add_argument("--upscale_factor", type=float, default=2.0)
    parser.add_argument("--merge_iou", type=float, default=0.5)
    args = parser.parse_args()

    from rfdetr import RFDETRSmall
    model = RFDETRSmall.from_checkpoint(args.checkpoint)
    print("Class names:", model.class_names)

    results = run_evaluation(model, args.dataset_dir, args.min_conf_for_views,
                              args.conf_thresholds, args.merge_iou, args.upscale_factor)

    print(f"\n=== TTA results on {args.dataset_dir} ===")
    print(f"(views: original + h-flip + {args.upscale_factor}x upscale, merge_iou={args.merge_iou})\n")
    for conf_thresh, r in results.items():
        print(f"--- final confidence threshold = {conf_thresh} ---")
        print(f"  Missed-violation rate: {r['missed_violation_rate']:.1%}")
        print(f"  False-alarm rate:      {r['false_alarm_rate']:.1%}")
        print(f"  Helmet precision: {r['helmet_precision']:.1%}  |  Helmet recall: {r['helmet_recall']:.1%}")
        for bucket in SCALE_LABELS:
            if bucket in r["per_scale_bucket_recall"]:
                print(f"    {bucket:>10}: {r['per_scale_bucket_recall'][bucket]:.1%}")
        print()


if __name__ == "__main__":
    main()
