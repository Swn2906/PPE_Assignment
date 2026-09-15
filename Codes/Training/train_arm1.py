"""
train_arm1.py

Trains Arm 1 (direct detection): RF-DETR Small on the 2-class
(head, helmet) dataset produced by prepare_arm1_data.py.

Reproduces the exact config used for the checkpoint reported in
RESULTS.md/DECISIONS.md. Uses RF-DETR's default augmentation
(aug_config=None -> horizontal flip, scale_jitter=True)

Usage:
    python train_arm1.py --dataset_dir Hard-Hat-Workers-10-2class --output_dir arm1_output
"""

import argparse

from rfdetr import RFDETRSmall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                         help="Output of prepare_arm1_data.py (2-class head/helmet dataset)")
    parser.add_argument("--output_dir", default="arm1_output")
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
    print("Use checkpoint_best_ema.pth for evaluation (evaluate_arm1.py) -- "
          "not checkpoint_best_regular.pth or the final-epoch weights.")


if __name__ == "__main__":
    main()
