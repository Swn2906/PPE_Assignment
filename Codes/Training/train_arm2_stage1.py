"""
train_arm2_stage1.py

Trains Arm 2's stage-1 detector: RF-DETR Small on the single-class
(headgear) dataset produced by prepare_arm2_stage1_data.py. This detector
only localizes head-regions -- the compliant/violation decision is made
separately by stage 2 (see train_crop_classifier.py).

Same epoch/batch/LR config as train_arm1.py, deliberately, so the two
arms are trained on a matched budget (see DECISIONS.md #4).

Usage:
    python train_arm2_stage1.py --dataset_dir Hard-Hat-Workers-10-headgear --output_dir arm2_stage1_output
"""

import argparse

from rfdetr import RFDETRSmall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                         help="Output of prepare_arm2_stage1_data.py (merged single-class headgear dataset)")
    parser.add_argument("--output_dir", default="arm2_stage1_output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="16 for A100-class GPUs; use 4 with grad_accum_steps=4 on a T4")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    args = parser.parse_args()

    model = RFDETRSmall()
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"\nTraining complete. Best EMA checkpoint saved under: {args.output_dir}")
    print("Next step: train the stage-2 classifier (extract_dinov2_features.py "
          "then train_crop_classifier.py) before running evaluate_arm2_pipeline.py.")


if __name__ == "__main__":
    main()
