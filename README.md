# PPE Compliance — Take-Home Submission

Two-arm comparison for hard-hat compliance detection on the Hard Hat
Workers v10 dataset. **Arm 1**: direct 2-class detection (RF-DETR).
**Arm 2**: cascade — class-agnostic detector (stage 1) + frozen-DINOv2
crop classifier (stage 2).

## Start here

| File | Contents |
|---|---|
| `Label_Policy.md` | Data audit findings, annotation spec, hard cases |
| `Eval_Plan.md` | Evaluation protocol, falsification criterion |
| `Decisions.md` | 9 key decisions, alternatives considered, two-more-weeks fixes |
| `Results.md` | The comparison, error analysis, recommendation |
| `figures/` | Charts referenced in RESULTS.md |

**Recommendation: Arm 1.** See RESULTS.md for the full comparison — Arm 2
does not recover the small-object/low-detail recall the cascade design
was meant to test for, at any configuration tried.

## Repo structure
```
Label_Policy.md
Eval_Plan.md
Decisions.md
Results.md
figures/
  scale_bucket_comparison.png
  threshold_comparison.png
Codes/
  Audit_Data_prep/   -- audit_utils.py, prepare_arm1_data.py, prepare_arm2_stage1_data.py
  Training/          -- train_arm1.py, train_arm2_stage1.py, build_crop_classifier_data.py,
                         extract_dinov2_features.py, train_crop_classifier.py
  Synthetic_data/    -- build_synthetic_small_scale.py
  Evaluation/        -- evaluate_arm1.py, evaluate_arm2_pipeline.py, evaluate_arm1_tta.py
  Visualization/     -- visualize_predictions.py, make_result_figures.py
Config_Metrics/
  Arm1/              -- saved training config + run log for Arm 1
  Arm2_Stage1/        -- saved training config + run log for Arm 2's stage-1 detector
requirements.txt
```

## Environment

- Google Colab, NVIDIA A100 (40GB)
- Python 3.13, `rfdetr==1.10.1`
- See `requirements.txt`

## Reproduction — run in this order

**1. Get the data**
```bash
pip install roboflow
# Roboflow "Hard Hat Workers" v10, joseph-nelson/hard-hat-workers
```

**2. Prep datasets**
```bash
python Codes/Audit_Data_prep/prepare_arm1_data.py --src Hard-Hat-Workers-10 --dst Hard-Hat-Workers-10-2class
python Codes/Audit_Data_prep/prepare_arm2_stage1_data.py --src Hard-Hat-Workers-10-2class --dst Hard-Hat-Workers-10-headgear
```

**3. Train both arms' detectors** (identical epoch/batch/LR budget — see DECISIONS.md #4)
```bash
python Codes/Training/train_arm1.py --dataset_dir Hard-Hat-Workers-10-2class --output_dir arm1_output
python Codes/Training/train_arm2_stage1.py --dataset_dir Hard-Hat-Workers-10-headgear --output_dir arm2_stage1_output
```

**4. Build and train Arm 2's stage-2 classifier**
```bash
python Codes/Training/build_crop_classifier_data.py --src Hard-Hat-Workers-10-2class --dst crop_classifier_data
python Codes/Training/extract_dinov2_features.py --data_dir crop_classifier_data --out_dir crop_features
python Codes/Training/train_crop_classifier.py --features_dir crop_features --out_path stage2_classifier.pt
```

**5. Build the resolution-degraded test slice** (see note in the script —
this tests reduced visual detail, not relative object-to-frame scale;
see Eval_Plan.md)
```bash
python Codes/Synthetic_data/build_synthetic_small_scale.py --src Hard-Hat-Workers-10-2class/test \
    --dst Hard-Hat-Workers-10-2class/test_resolution_degraded --target_rel_h 0.021
```

**6. Evaluate**
```bash
python Codes/Evaluation/evaluate_arm1.py --checkpoint arm1_output/checkpoint_best_ema.pth \
    --dataset_dir Hard-Hat-Workers-10-2class/test
python Codes/Evaluation/evaluate_arm2_pipeline.py --stage1_checkpoint arm2_stage1_output/checkpoint_best_ema.pth \
    --stage2_checkpoint stage2_classifier.pt --dataset_dir Hard-Hat-Workers-10-2class/test
```
Repeat both against `test_resolution_degraded` for the supplementary
slice. `Codes/Evaluation/evaluate_arm1_tta.py` runs the TTA side
experiment (Decisions.md #8).

**7. Figures and qualitative comparison**
```bash
python Codes/Visualization/make_result_figures.py --out_dir figures
python Codes/Visualization/visualize_predictions.py --dataset_dir Hard-Hat-Workers-10-2class/test \
    --arm1_checkpoint arm1_output/checkpoint_best_ema.pth \
    --stage1_checkpoint arm2_stage1_output/checkpoint_best_ema.pth \
    --stage2_checkpoint stage2_classifier.pt
```

## Not included
Trained checkpoints (`.pth`/`.pt` files) are excluded from this repo —
reproducible via the training scripts above, not committed as large
binaries. `Codes/Audit_Data_prep/audit_utils.py` holds shared utilities
used throughout data auditing and visualization.
