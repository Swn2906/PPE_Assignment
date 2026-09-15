"""
visualize_predictions.py

Shows a 3-panel comparison per sampled test image: ground truth, Arm 1's
predictions, and Arm 2's (stage-1 + stage-2 cascade) predictions -- for
qualitative error analysis in RESULTS.md.

Usage:
    python visualize_predictions.py \
        --dataset_dir Hard-Hat-Workers-10-2class/test \
        --arm1_checkpoint arm1_output/checkpoint_best_ema.pth \
        --stage1_checkpoint arm2_stage1_output/checkpoint_best_ema.pth \
        --stage2_checkpoint stage2_classifier.pt \
        --n_samples 6 \
        --conf_thresh 0.5 \
        --out_path predictions_comparison.png
"""

import argparse
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


COLORS_GT = {"head": (255, 0, 0), "helmet": (0, 255, 0)}          # red=violation, green=compliant
COLORS_PRED = {"head": (255, 140, 0), "helmet": (0, 200, 200)}    # distinct shades for predictions


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def draw_boxes(img, boxes, thickness=2):
    """boxes: list of (x1, y1, x2, y2, class_name, color, label_text)"""
    img = img.copy()
    for x1, y1, x2, y2, cls, color, label in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, label, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def get_gt_boxes(anns, cat_by_id):
    boxes = []
    for a in anns:
        name = cat_by_id[a["category_id"]]
        x, y, w, h = a["bbox"]
        boxes.append((x, y, x + w, y + h, name, COLORS_GT.get(name, (255, 255, 0)), name))
    return boxes


def get_arm1_boxes(arm1_model, img_path, conf_thresh):
    dets = arm1_model.predict(img_path, threshold=conf_thresh)
    boxes = []
    for i in range(len(dets.xyxy)):
        x1, y1, x2, y2 = dets.xyxy[i]
        name = arm1_model.class_names[int(dets.class_id[i])]
        conf = dets.confidence[i]
        boxes.append((x1, y1, x2, y2, name, COLORS_PRED.get(name, (255, 255, 0)), f"{name} {conf:.2f}"))
    return boxes


def classify_crop(crop_bgr, processor, backbone, head, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = backbone(**inputs).last_hidden_state[:, 0, :]
        logits = head(feat)
        prob_violation = torch.softmax(logits, dim=1)[0, 1].item()
    return prob_violation


def get_arm2_boxes(stage1_model, processor, backbone, head, device, img, img_path,
                    stage1_conf, stage2_thresh, pad_frac=0.25):
    img_h, img_w = img.shape[:2]
    dets = stage1_model.predict(img_path, threshold=stage1_conf)
    boxes = []
    for i in range(len(dets.xyxy)):
        x1, y1, x2, y2 = dets.xyxy[i]
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * pad_frac, h * pad_frac
        cx1 = max(0, int(x1 - pad_x)); cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x)); cy2 = min(img_h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = img[cy1:cy2, cx1:cx2]
        stage1_conf_score = float(dets.confidence[i])
        prob_violation = classify_crop(crop, processor, backbone, head, device)
        name = "head" if prob_violation >= stage2_thresh else "helmet"
        # show the probability of the PREDICTED class, so it reads naturally
        # either way (e.g. "helmet 0.88" not "helmet 0.12"), plus stage-1's
        # own detection confidence for that box, since the two are different
        # things (localization+existence vs. crop classification).
        shown_prob = prob_violation if name == "head" else (1 - prob_violation)
        label = f"{name} s1={stage1_conf_score:.2f} s2={shown_prob:.2f}"
        boxes.append((x1, y1, x2, y2, name, COLORS_PRED.get(name, (255, 255, 0)), label))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--arm1_checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", default="predictions_comparison.png")
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.dataset_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    from rfdetr import RFDETRSmall
    arm1_model = RFDETRSmall.from_checkpoint(args.arm1_checkpoint)
    stage1_model = RFDETRSmall.from_checkpoint(args.stage1_checkpoint)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    backbone = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    stage2_head = MLPHead(in_dim=384).to(device)
    stage2_head.load_state_dict(torch.load(args.stage2_checkpoint, map_location=device))
    stage2_head.eval()

    candidates = [img for img in coco["images"] if by_image.get(img["id"])]
    sample = random.sample(candidates, min(args.n_samples, len(candidates)))

    fig, axes = plt.subplots(len(sample), 3, figsize=(15, 5 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for row, img_info in zip(axes, sample):
        img_path = os.path.join(args.dataset_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_boxes = get_gt_boxes(by_image[img_info["id"]], cat_by_id)
        arm1_boxes = get_arm1_boxes(arm1_model, img_path, args.conf_thresh)
        arm2_boxes = get_arm2_boxes(stage1_model, processor, backbone, stage2_head, device,
                                     img, img_path, args.conf_thresh, args.conf_thresh)

        row[0].imshow(draw_boxes(img_rgb, gt_boxes))
        row[0].set_title(f"Ground truth\n{img_info['file_name']}", fontsize=8)
        row[0].axis("off")

        row[1].imshow(draw_boxes(img_rgb, arm1_boxes))
        row[1].set_title("Arm 1 (direct detection)", fontsize=8)
        row[1].axis("off")

        row[2].imshow(draw_boxes(img_rgb, arm2_boxes))
        row[2].set_title("Arm 2 (cascade)", fontsize=8)
        row[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=100)
    print(f"Saved comparison grid to {args.out_path}")


if __name__ == "__main__":
    main()
