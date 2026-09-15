"""
make_result_figures.py

Generates the two charts referenced in RESULTS.md:
  1. threshold_comparison.png -- missed-violation & false-alarm rate,
     Arm 1 vs Arm 2, across the confidence threshold sweep (natural test set).
  2. scale_bucket_comparison.png -- recall by object scale bucket,
     Arm 1 vs Arm 2, at the chosen operating point (threshold=0.5).

Run this after both evaluate_arm1.py and evaluate_arm2_pipeline.py have
been run on the natural test set, with their printed numbers hard-coded
below (kept as plain numbers here for simplicity/reproducibility --
no live model inference needed to make these plots).

Usage:
    python make_result_figures.py --out_dir figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def make_threshold_chart(out_path):
    thresholds = [0.3, 0.5, 0.7, 0.9]
    arm1_missed = [2.5, 4.7, 11.8, 91.5]
    arm1_false_alarm = [8.8, 5.3, 2.9, 0.0]
    arm2_missed = [6.1, 6.3, 6.6, 7.3]
    arm2_false_alarm = [6.1, 5.9, 5.4, 4.9]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(thresholds, arm1_missed, marker="o", label="Arm 1")
    axes[0].plot(thresholds, arm2_missed, marker="s", label="Arm 2")
    axes[0].set_title("Missed-violation rate")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Missed-violation rate (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, arm1_false_alarm, marker="o", label="Arm 1")
    axes[1].plot(thresholds, arm2_false_alarm, marker="s", label="Arm 2")
    axes[1].set_title("False-alarm rate")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("False-alarm rate (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def make_scale_bucket_chart(out_path):
    buckets = ["<15px\n(n=4)", "15-30px\n(n=233)", "30-60px\n(n=410)", "60-100px\n(n=71)", ">100px\n(n=8)"]
    arm1_recall = [100.0, 90.3, 97.2, 100.0, 87.5]
    arm2_recall = [50.0, 89.4, 95.8, 95.8, 87.5]

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, arm1_recall, width, label="Arm 1")
    ax.bar(x + width/2, arm2_recall, width, label="Arm 2")

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall by scale bucket, natural test set (threshold=0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_threshold_chart(os.path.join(args.out_dir, "threshold_comparison.png"))
    make_scale_bucket_chart(os.path.join(args.out_dir, "scale_bucket_comparison.png"))


if __name__ == "__main__":
    main()
