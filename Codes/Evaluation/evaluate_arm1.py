"""
evaluate_arm1.py

Evaluates a trained RF-DETR checkpoint (Arm 1, direct detection) against a
COCO-format test set, reporting:
  - Per-scale-bucket recall (missed-violation rate for the `head` class)
  - Overall precision/false-alarm rate at a chosen confidence threshold
  - A small confidence-threshold sweep, since the assignment asks you to
    choose and justify an operating point (favoring precision, since false
    positives are more expensive than false negatives per the brief)

This uses greedy, class-aware IoU matching (standard single-threshold
precision/recall, not full COCO mAP) since the assignment wants an
operating-point business metric, not a headline mAP number.

Usage (inside Colab, after `from rfdetr import RFDETRSmall`):
    python evaluate_arm1.py \
        --checkpoint arm1_output/checkpoint_best_ema.pth \
        --dataset_dir Hard-Hat-Workers-10-2class/test \
        --model_size small \
        --iou_thresh 0.5 \
        --conf_thresholds 0.3 0.5 0.7 0.9
"""

import argparse
import json
import os
from collections import defaultdict

import cv2


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

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


def load_model(model_size, checkpoint_path):
    import rfdetr as rfdetr_module
    cls = getattr(rfdetr_module, MODEL_CLASSES[model_size])
    model = cls.from_checkpoint(checkpoint_path)
    print("Loaded model. Class names:", model.class_names)
    return model


def run_evaluation(model, dataset_dir, iou_thresh, conf_thresholds):
    with open(os.path.join(dataset_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    # class_names on the loaded model may be 0-indexed; build a name->name
    # passthrough and rely on matching by STRING NAME, not id, to avoid
    # off-by-one indexing mismatches between training categories and
    # inference class_id outputs. Confirm this mapping manually against the
    # printed model.class_names before trusting results.
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    results = {}

    for conf_thresh in conf_thresholds:
        # per-scale-bucket: [tp, fn] for the `head` (violation) class
        violation_bucket_stats = defaultdict(lambda: {"tp": 0, "fn": 0})
        # overall counts for false-alarm-rate / precision, `head` class only
        head_tp, head_fp = 0, 0
        # also track helmet class overall for completeness
        helmet_tp, helmet_fp, helmet_fn = 0, 0, 0

        for img_info in coco["images"]:
            img_path = os.path.join(dataset_dir, img_info["file_name"])
            anns = gt_by_image.get(img_info["id"], [])

            gts = []
            for a in anns:
                name = cat_by_id[a["category_id"]]
                gts.append({
                    "bbox": to_xyxy(a["bbox"]),
                    "class": name,
                    "h": a["bbox"][3],
                    "matched": False,
                })

            detections = model.predict(img_path, threshold=conf_thresh)
            # detections: sv.Detections-like object with .xyxy, .class_id, .confidence
            preds = []
            for i in range(len(detections.xyxy)):
                class_id = int(detections.class_id[i])
                # model.class_names is expected to be indexable by class_id
                name = model.class_names[class_id] if model.class_names else str(class_id)
                preds.append({
                    "bbox": tuple(detections.xyxy[i]),
                    "class": name,
                    "conf": float(detections.confidence[i]),
                })

            # sort predictions by confidence descending for greedy matching
            preds.sort(key=lambda p: -p["conf"])

            for p in preds:
                # find best unmatched GT of the SAME class with IoU >= thresh
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
                    if is_tp:
                        head_tp += 1
                    else:
                        head_fp += 1
                elif p["class"] == "helmet":
                    if is_tp:
                        helmet_tp += 1
                    else:
                        helmet_fp += 1

            # unmatched GTs are misses (missed violations, if class == head)
            for g in gts:
                if g["class"] == "head":
                    bucket = scale_bucket(g["h"])
                    if g["matched"]:
                        violation_bucket_stats[bucket]["tp"] += 1
                    else:
                        violation_bucket_stats[bucket]["fn"] += 1
                elif g["class"] == "helmet" and not g["matched"]:
                    helmet_fn += 1

        # ---- assemble this threshold's summary ----
        total_head_tp = sum(v["tp"] for v in violation_bucket_stats.values())
        total_head_fn = sum(v["fn"] for v in violation_bucket_stats.values())
        missed_violation_rate = (
            total_head_fn / (total_head_tp + total_head_fn)
            if (total_head_tp + total_head_fn) else float("nan")
        )
        false_alarm_rate = (
            head_fp / (head_tp + head_fp) if (head_tp + head_fp) else float("nan")
        )

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
    parser.add_argument("--model_size", default="small", choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--conf_thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()

    model = load_model(args.model_size, args.checkpoint)
    results = run_evaluation(model, args.dataset_dir, args.iou_thresh, args.conf_thresholds)

    print(f"\n=== Results on {args.dataset_dir} (IoU thresh = {args.iou_thresh}) ===\n")
    for conf_thresh, r in results.items():
        print(f"--- confidence threshold = {conf_thresh} ---")
        print(f"  Missed-violation rate (head recall miss): {r['missed_violation_rate']:.1%}")
        print(f"  False-alarm rate (head precision miss):   {r['false_alarm_rate']:.1%}")
        print(f"  Helmet precision: {r['helmet_precision']:.1%}  |  Helmet recall: {r['helmet_recall']:.1%}")
        print(f"  Per-scale-bucket head recall:")
        for bucket in SCALE_LABELS:
            if bucket in r["per_scale_bucket_recall"]:
                print(f"    {bucket:>10}: {r['per_scale_bucket_recall'][bucket]:.1%}")
        print()


if __name__ == "__main__":
    main()
