**Operating Point:**   
**Confidence=0.5, IoU=0.5** (fixed matching threshold; see DECISIONS.md \#5). Chosen as the balance point before recall (0.3) trades too much false-alarm, or precision (0.7-0.9) trades too much recall — see full sweep below.  
**Natural test set (primary evidence)**

| Threshold | Arm1 MVR | Arm1 FAR | Arm2 MVR | Arm2 FAR |
| :---: | :---: | :---: | :---: | :---: |
| 0.3 | 2.5%  | 8.8%  | 6.1%  | 6.1%  |
| 0.5 | 4.7%  | 5.3%  | 6.3%  | 5.9%  |
| 0.7 | 11.8% | 2.9% | 6.6% | 5.4% |
| 0.9 | 91.5% | 0.0% | 7.3% | 4.9% |

Note: “Threshold” refers to different cutoffs for each arm. For Arm 1, it is the detector’s confidence threshold, affecting both box generation and classification. For Arm 2, it refers only to the Stage 2 classification threshold, while Stage 1 is fixed at 0.5 confidence. Therefore, Stage 1’s recall ceiling remains unchanged across these results.  
**Scale-bucket recall (threshold=0.5)**

| Bucket | n (real head boxes)  | Arm 1 recall  | Arm 2 recall  |
| :---: | :---: | :---: | :---: |
| \<15px  | 4 | 100.0% | 50.0% |
| 15-30px  | 233 | 90.3% | 89.4% |
| 30-60px  | 410 | 97.2% | 95.8% |
| 60-100px  | 71 | 100% | 95.8% |
| \>100px  | 8 | 87.5% | 87.5% |

Buckets with \<15px (n=4) and \>100px (n=8) are too small for reliable conclusions and should be treated as directional. The well-populated 15–100px buckets (n=233–410) provide the meaningful comparison, where Arm 1 consistently outperformed Arm 2\.  
Helmet precision/recall remained high (92–98%+) across both arms, so the main weakness is concentrated in the head/violation class, not overall detection. We therefore focused scale-wise analysis on recall, where the failure was most evident.

**Resolution-degraded slice (supplementary, not deployment-predictive)**

| Threshold | Arm1 MVR | ARM2 MVR |
| :---: | :---: | :---: |
| **0.3** | **63.1%** | **84.0%** |
| **0.5** | **74.2%** | **84.4%** |
| **0.7** | **88.2%** | **85.3%** |
| **0.9** | **100%** | **86.1%** |

Stage 1 achieved only 22.0% recall on this slice, nearly matching Arm 2’s end-to-end result. Since Stage 2 achieves 97%+ on ground-truth crops, this confirms that the main bottleneck is Stage 1 localization, not classification.  
**Why Arm 2 looks stable across its own sweep — and doesn't beat Arm 1 anyway**  
The sweep above only varies stage-2's cutoff at one fixed stage-1 confidence (0.5) — stage-1's recall ceiling never moves. Re-run with a looser stage-1 (0.3) to check this wasn't just an unlucky setting:

| Config | MVR | FAR |
| :---: | :---: | :---: |
| Arm 2, stage1=0.5, stage2=0.5 | 6.3% | 5.9% |
| Arm 2, stage1=0.3, stage2=0.3  | 4.1% | 9.8% |

Loosening stage-1 trades recall for precision — it doesn't fix the architecture.

**TTA (side experiment, Arm1)**

| Threshold | Natural: no-TTA / TTA MVR | Degraded: no-TTA / TTAMVR | Natural: no TTA/TTAFAR |
| :---: | :---: | :---: | :---: |
| 0.3 | 2.5% / 5.0% | 63.1% / 87.5%  | 8.8%/8.4% |
| 0.5 | 4.7% / 7.4%  | 74.2% / 95.5% | 5.3%/ 3.6% |

TTA showed a small trade-off on the natural test set, but caused a significant drop in recall for small objects. The likely reason is inaccurate box fusion at very small box sizes, so TTA was not adopted.

**Where Each Arm Fails?**

* Arm 1: Struggles most in low-detail/crowded scenes and on the minority head class (\~25% of instances), likely influenced by the 3:1 helmet-to-head imbalance.   
* Arm 2: Inherits the same Stage 1 weakness and adds another failure point: Stage 2 cannot classify a crop that Stage 1 fails to detect. The hypothesis that decoupling would improve degraded-input recall was therefore not supported.


**My Recommendation:**  
**Arm 1**. Arm 2 adds latency and another failure point without showing a measurable benefit on either test condition across the configurations tested. Neither arm is deployment-ready for the target scale/detail regime, indicating a broader data or architectural gap that the cascade does not resolve.

**Where the Tool Got It Wrong, and How We Caught It**  
Our first synthetic test was intended to simulate small objects, but shrinking the image and boxes together preserved their relative scale. We caught this by comparing the box-height distributions before and after transformation—the histograms overlapped instead of shifting.

We therefore reframed the test as a resolution-degradation test rather than a small-object test.