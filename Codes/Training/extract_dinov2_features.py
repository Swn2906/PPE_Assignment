"""
extract_dinov2_features.py

Extracts frozen DINOv2 (ViT) embeddings for every crop in the
compliant/violation crop-classifier dataset, and caches them as tensors
per split. This lets the actual classifier training step (a small MLP) run
in seconds, since the expensive backbone forward pass happens only once.

Usage:
    python extract_dinov2_features.py \
        --data_dir crop_classifier_data \
        --out_dir crop_features \
        --model facebook/dinov2-small
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

CLASSES = ["compliant", "violation"]  # compliant=0, violation=1


def extract_split(data_dir, split, processor, model, device):
    features, labels = [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, split, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(os.listdir(class_dir))

        for fname in tqdm(files, desc=f"{split}/{class_name}"):
            img_path = os.path.join(class_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable image {img_path}: {e}")
                continue

            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token embedding (first token of last_hidden_state)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

            features.append(cls_embedding)
            labels.append(class_idx)

    if not features:
        return None, None

    return torch.stack(features), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    for split in ["train", "valid", "test"]:
        feats, labels = extract_split(args.data_dir, split, processor, model, device)
        if feats is None:
            print(f"[{split}] no data found, skipping")
            continue
        torch.save({"features": feats, "labels": labels}, os.path.join(args.out_dir, f"{split}.pt"))
        print(f"[{split}] saved {feats.shape[0]} embeddings, dim={feats.shape[1]} "
              f"(compliant={int((labels==0).sum())}, violation={int((labels==1).sum())})")


if __name__ == "__main__":
    main()
