\# PPE Compliance — Take-Home Submission



Two-arm comparison for hard-hat compliance detection on the Hard Hat

Workers v10 dataset. \*\*Arm 1\*\*: direct 2-class detection (RF-DETR).

\*\*Arm 2\*\*: cascade — class-agnostic detector (stage 1) + frozen-DINOv2

crop classifier (stage 2).



\## Start here



| File | Contents |

|---|---|

| `LABEL\_POLICY.md` | Data audit findings, annotation spec, hard cases |

| `EVAL\_PLAN.md` | Evaluation protocol, falsification criterion |

| `DECISIONS.md` | 9 key decisions, alternatives considered, two-more-weeks fixes |

| `RESULTS.md` | The comparison, error analysis, recommendation |

| `figures/` | Charts referenced in RESULTS.md |



\*\*Recommendation: Arm 1.\*\* See RESULTS.md for the full comparison — Arm 2

does not recover the small-object/low-detail recall the cascade design

was meant to test for, at any configuration tried.



\## Repo structure

```

LABEL\_POLICY.md

EVAL\_PLAN.md

DECISIONS.md

RESULTS.md

figures/

&#x20; scale\_bucket\_comparison.png

&#x20; threshold\_comparison.png

Codes/

&#x20; Audit\_Data\_prep/   -- audit\_utils.py, prepare\_arm1\_data.py, prepare\_arm2\_stage1\_data.py

&#x20; Training/          -- train\_arm1.py, train\_arm2\_stage1.py, build\_crop\_classifier\_data.py,

&#x20;                        extract\_dinov2\_features.py, train\_crop\_classifier.py

&#x20; Synthetic\_data/    -- build\_synthetic\_small\_scale.py, augment\_train\_scale.py (incomplete)

&#x20; Evaluation/        -- evaluate\_arm1.py, evaluate\_arm2\_pipeline.py, evaluate\_arm1\_tta.py

&#x20; Visualization/     -- visualize\_predictions.py, make\_result\_figures.py

Config\_Metrics/

&#x20; Arm1/              -- saved training config + run log for Arm 1

&#x20; Arm2\_Stage1/        -- saved training config + run log for Arm 2's stage-1 detector

requirements.txt

```



\## Environment



\- Google Colab, NVIDIA A100 (40GB)

\- Python 3.13, `rfdetr==1.10.1`

\- See `requirements.txt`



\## Reproduction — run in this order



\*\*1. Get the data\*\*

```bash

pip install roboflow

\# Roboflow "Hard Hat Workers" v10, joseph-nelson/hard-hat-workers

```



\*\*2. Prep datasets\*\*

```bash

python Codes/Audit\_Data\_prep/prepare\_arm1\_data.py --src Hard-Hat-Workers-10 --dst Hard-Hat-Workers-10-2class

python Codes/Audit\_Data\_prep/prepare\_arm2\_stage1\_data.py --src Hard-Hat-Workers-10-2class --dst Hard-Hat-Workers-10-headgear

```



\*\*3. Train both arms' detectors\*\* (identical epoch/batch/LR budget — see DECISIONS.md #4)

```bash

python Codes/Training/train\_arm1.py --dataset\_dir Hard-Hat-Workers-10-2class --output\_dir arm1\_output

python Codes/Training/train\_arm2\_stage1.py --dataset\_dir Hard-Hat-Workers-10-headgear --output\_dir arm2\_stage1\_output

```



\*\*4. Build and train Arm 2's stage-2 classifier\*\*

```bash

python Codes/Training/build\_crop\_classifier\_data.py --src Hard-Hat-Workers-10-2class --dst crop\_classifier\_data

python Codes/Training/extract\_dinov2\_features.py --data\_dir crop\_classifier\_data --out\_dir crop\_features

python Codes/Training/train\_crop\_classifier.py --features\_dir crop\_features --out\_path stage2\_classifier.pt

```



\*\*5. Build the resolution-degraded test slice\*\* (see note in the script —

this tests reduced visual detail, not relative object-to-frame scale;

see EVAL\_PLAN.md)

```bash

python Codes/Synthetic\_data/build\_synthetic\_small\_scale.py --src Hard-Hat-Workers-10-2class/test \\

&#x20;   --dst Hard-Hat-Workers-10-2class/test\_resolution\_degraded --target\_rel\_h 0.021

```



\*\*6. Evaluate\*\*

```bash

python Codes/Evaluation/evaluate\_arm1.py --checkpoint arm1\_output/checkpoint\_best\_ema.pth \\

&#x20;   --dataset\_dir Hard-Hat-Workers-10-2class/test

python Codes/Evaluation/evaluate\_arm2\_pipeline.py --stage1\_checkpoint arm2\_stage1\_output/checkpoint\_best\_ema.pth \\

&#x20;   --stage2\_checkpoint stage2\_classifier.pt --dataset\_dir Hard-Hat-Workers-10-2class/test

```

Repeat both against `test\_resolution\_degraded` for the supplementary

slice. `Codes/Evaluation/evaluate\_arm1\_tta.py` runs the TTA side

experiment (DECISIONS.md #8).



\*\*7. Figures and qualitative comparison\*\*

```bash

python Codes/Visualization/make\_result\_figures.py --out\_dir figures

python Codes/Visualization/visualize\_predictions.py --dataset\_dir Hard-Hat-Workers-10-2class/test \\

&#x20;   --arm1\_checkpoint arm1\_output/checkpoint\_best\_ema.pth \\

&#x20;   --stage1\_checkpoint arm2\_stage1\_output/checkpoint\_best\_ema.pth \\

&#x20;   --stage2\_checkpoint stage2\_classifier.pt

```



\## Not included

Trained checkpoints (`.pth`/`.pt` files) are excluded from this repo —

reproducible via the training scripts above, not committed as large

binaries. `Codes/Audit\_Data\_prep/audit\_utils.py` holds shared utilities

used throughout data auditing and visualization

