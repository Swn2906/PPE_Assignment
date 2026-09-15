**Metrics**

Two business metrics, not mAP: **missed-violation rate** (real head instances with no matching detection) and **false-alarm rate** (predicted head instances with no matching ground truth). A match requires the same class and IoU ≥ 0.5. A tighter IoU threshold doesn't selectively punish bad detections — it converts imprecise-but-real detections into both a false alarm and a missed violation simultaneously, so 0.5 is kept fixed rather than tuned.

Both metrics are reported across a confidence-threshold sweep (0.3, 0.5, 0.7, and 0.9), as the threshold represents the key trade-off available at deployment. Given that false positives (FPs) are more costly than false negatives (FNs) in this application, the selected operating point needs to be justified accordingly.

**Two Test Conditions:**

**Natural test set:** 706 untouched held-out images, used for all primary metrics. It is dominated by larger heads (median \= 10.4% of frame height), with only 0.3% matching the deployment target of \~1.4–2.8%.  
**Synthetic stress test:** Images and boxes are jointly downscaled to simulate reduced visual detail while preserving composition. This provides a **directional check** of performance on smaller targets, not a direct estimate of deployment accuracy.

**Comparability Across Arms**

All arms and side experiments use the **same test images, matching code, IoU threshold, and metric definitions** for a fair comparison.

**Arm 2** is evaluated as the full end-to-end cascade (Stage 1 → crop → Stage 2), while Stage 2 on ground-truth crops is reported separately as a **diagnostic only**.

**What would falsify the core hypothesis** 

Arm 2 exists to test whether decoupling localization from classification recovers recall lost by a single joint detector on harder, lower-detail inputs. **Falsified if Arm 2's missed-violation rate on the natural test set (and the supplementary synthetic slice) is not meaningfully better than Arm 1's at matched thresholds.** Secondary diagnostic: if stage-1 alone (ignoring stage-2) shows the same recall pattern as Arm 1, that points to shared architecture/training limitations rather than anything cascade-specific. 

