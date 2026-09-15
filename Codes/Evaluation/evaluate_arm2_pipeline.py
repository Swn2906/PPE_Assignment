"""
evaluate_arm2_pipeline.py

Runs Arm 2's FULL cascade end-to-end on real images (not ground-truth
crops): stage-1 detector proposes `headgear` boxes on the raw image, each
box is cropped (same padding convention as training) and classified by
stage-2's frozen-DINOv2 + MLP head, and the resulting (box, label) pairs are
compared against ground truth using the identical missed-violation-rate /
false-alarm-rate / scale-bucket framing as evaluate_arm1.py, so the two arms
are directly comparable under the same conditions.

Usage:
    python evaluate_arm2_pipeline.py \
        --stage1_checkpoint arm2_stage1_output/checkpoint_best_ema.pth \
        --stage2_checkpoint stage2_classifier.pt \
        --dataset_dir Hard-Hat-Workers-10-2class/test \
        --stage1_conf 0.5 \
        --stage2_thresholds 0.3 0.5 0.7 0.9 \
        --pad_frac 0.25
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


SCALE_BINS = [0, 15, 30, 60, 100, 10000]
SCALE_LABELS = ["<15px", "15-30px", "30-60px", "60-100px", ">100px"]


class MLPHead(nn.Module):
    """Must match train_crop_classifier.py's architecture exactly."""
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def load_stage1(checkpoint_path):
    from rfdetr import RFDETRSmall
    model = RFDETRSmall.from_checkpoint(checkpoint_path)
    print("Stage-1 class names:", model.class_names)
    return model


def load_stage2(checkpoint_path, device, dinov2_name="facebook/dinov2-small"):
    processor = AutoImageProcessor.from_pretrained(dinov2_name)
    backbone = AutoModel.from_pretrained(dinov2_name).to(device).eval()
    head = MLPHead(in_dim=384).to(device)
    head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    head.eval()
    return processor, backbone, head


def classify_crop(crop_bgr, processor, backbone, head, device):
    """Returns P(violation) for a single crop (numpy BGR array)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = backbone(**inputs).last_hidden_state[:, 0, :]
        logits = head(feat)
        prob_violation = torch.softmax(logits, dim=1)[0, 1].item()
    return prob_violation


def run_pipeline_evaluation(stage1_model, processor, backbone, head, device,
                             dataset_dir, stage1_conf, stage2_thresholds,
                             iou_thresh, pad_frac):
    with open(os.path.join(dataset_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    # Run stage 1 + stage 2 ONCE per image at a fixed stage1_conf, caching the
    # resulting (box, prob_violation) pairs -- then sweep stage2 thresholds
    # cheaply without re-running inference for each one.
    cached_predictions = {}  # image_id -> list of (xyxy, prob_violation)

    for img_info in coco["images"]:
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        detections = stage1_model.predict(img_path, threshold=stage1_conf)
        preds = []
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = w * pad_frac, h * pad_frac
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(img_w, int(x2 + pad_x))
            cy2 = min(img_h, int(y2 + pad_y))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = img[cy1:cy2, cx1:cx2]
            prob_violation = classify_crop(crop, processor, backbone, head, device)
            preds.append(((x1, y1, x2, y2), prob_violation))

        cached_predictions[img_info["id"]] = preds

    results = {}
    for stage2_thresh in stage2_thresholds:
        violation_bucket_stats = defaultdict(lambda: {"tp": 0, "fn": 0})
        head_tp, head_fp = 0, 0
        helmet_tp, helmet_fp, helmet_fn = 0, 0, 0

        for img_info in coco["images"]:
            anns = gt_by_image.get(img_info["id"], [])
            gts = []
            for a in anns:
                name = cat_by_id[a["category_id"]]
                gts.append({"bbox": to_xyxy(a["bbox"]), "class": name,
                            "h": a["bbox"][3], "matched": False})

            preds = cached_predictions[img_info["id"]]
            labeled_preds = [
                {"bbox": box, "class": "head" if p >= stage2_thresh else "helmet", "conf": p}
                for box, p in preds
            ]
            labeled_preds.sort(key=lambda p: -p["conf"] if p["class"] == "head" else p["conf"])

            for p in labeled_preds:
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
                else:
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

        results[stage2_thresh] = {
            "missed_violation_rate": missed_violation_rate,
            "false_alarm_rate": false_alarm_rate,
            "per_scale_bucket_recall": per_bucket_recall,
            "helmet_precision": helmet_tp / (helmet_tp + helmet_fp) if (helmet_tp + helmet_fp) else float("nan"),
            "helmet_recall": helmet_tp / (helmet_tp + helmet_fn) if (helmet_tp + helmet_fn) else float("nan"),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--stage1_conf", type=float, default=0.5)
    parser.add_argument("--stage2_thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--pad_frac", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model = load_stage1(args.stage1_checkpoint)
    processor, backbone, head = load_stage2(args.stage2_checkpoint, device)

    results = run_pipeline_evaluation(
        stage1_model, processor, backbone, head, device,
        args.dataset_dir, args.stage1_conf, args.stage2_thresholds,
        args.iou_thresh, args.pad_frac,
    )

    print(f"\n=== Arm 2 (cascade) results on {args.dataset_dir} ===")
    print(f"(stage1_conf={args.stage1_conf}, iou_thresh={args.iou_thresh})\n")
    for stage2_thresh, r in results.items():
        print(f"--- stage2 violation threshold = {stage2_thresh} ---")
        print(f"  Missed-violation rate: {r['missed_violation_rate']:.1%}")
        print(f"  False-alarm rate:      {r['false_alarm_rate']:.1%}")
        print(f"  Helmet precision: {r['helmet_precision']:.1%}  |  Helmet recall: {r['helmet_recall']:.1%}")
        print(f"  Per-scale-bucket head recall:")
        for bucket in SCALE_LABELS:
            if bucket in r["per_scale_bucket_recall"]:
                print(f"    {bucket:>10}: {r['per_scale_bucket_recall'][bucket]:.1%}")
        print()


if __name__ == "__main__":
    main()
