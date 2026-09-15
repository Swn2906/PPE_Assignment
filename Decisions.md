## **1\. Backbone/model: RF-DETR (DINOv2 ViT)**

We considered RF-DETR, Deformable DETR, Swin \+ Cascade R-CNN, YOLOS, and ViTDet. We selected RF-DETR with a DINOv2 (ViT-family) backbone because it satisfies the brief, offers real-time inference suitable for the single-GPU/12-stream deployment constraint, and converged quickly on our relatively small custom dataset.

On the natural test set, RF-DETR performed reasonably well on the target 15–30px head sizes, achieving 79–95% recall across confidence thresholds of 0.3–0.7. Performance dropped more noticeably on the synthetic resolution-degraded subset, which we use as a stress test. This likely reflects the reduced visual detail and limited representation of similarly degraded examples in the training data.

Given the assignment timeline, we kept the architecture fixed and focused on evaluating its performance and failure modes rather than changing models mid-assignment. With more time, we would compare alternative architectures and add more degraded/low-detail examples to improve robustness.

## **2\. person Dropped; head Used as the Detection Proxy**

Options: Keep person label despite its sparsity, train with it as an additional target, or drop it and use head as the proxy for a person.

Chose: We dropped person since it made up only 2.2 \-- 2.4% of labels and was biased toward close-range images(which is not the deployment target). The designated area is instead enforced using a fixed ROI polygon on detected head/helmet boxes, so a separate person detector is not needed.

Known limitation: At inference time, if the detector completely misses a person's head/helmet, the system produces no verdict for that person. Without an independent person-presence signal, this becomes a silent miss rather than an "uncertain" case that could be routed for review.

If we had more time: We would evaluate additional person-detection approaches and add a "needs review" state to handle these cases.

**3\. Cascade Design: Merge Classes at Stage 1**

Options: Keep separate head and helmet classes in Stage 1, or merge them into a single headgear class and let Stage 2 handle the final classification.

Chose: We merged them into a single class to simplify Stage 1 into a pure localization task and test whether decoupling detection and classification would improve recall.

What we found: It didn't. Stage 1 showed nearly the same recall drop on the resolution-degraded slice as Arm 1, suggesting the bottleneck lies in the shared architecture rather than the task design.

**4\. Matched Training Budget Across Arms**

Options: Tune each arm independently or use the same training budget.

Chose: We used the same 15 epochs, batch size, and learning-rate configuration for both Arm 1 and Arm 2 Stage 1\. This keeps the comparison focused on the architecture and design, rather than giving one approach more compute.

**5\. IoU Fixed at 0.5; Confidence Threshold as the Main Lever**

Options: Tune the IoU threshold, confidence threshold, or both.

Chose: We fixed IoU at 0.5 as the standard evaluation threshold and swept the confidence threshold from 0.3–0.9. Confidence is the more relevant deployment lever here, as it directly controls the trade-off between missed violations and false alarms, with false positives being more costly.

**6\. Synthetic Testing for the Small-Object Regime**

Goal: Approximate the deployment setting, since \<0.3% of the natural test set contains objects in the target small-object range.

What we did: Created a synthetic slice by shrinking the full image and its bounding boxes together to test performance under reduced visual detail. We use this as a supplementary, directional signal, while the natural test set remains the primary evaluation.

Would change our mind if: We had genuine distant-camera footage. Real deployment data would be preferable to further synthetic approximation. 

**7\.  Default Augmentation**

We used RF-DETR's default augmentation (aug\_config: null, scale\_jitter: true), which included horizontal flips and random resize/crop. These were active throughout training but were not specifically tuned for the deployment scenario.

If we had more time: We would use a custom augmentation strategy focused on small and distant objects, rather than relying on the defaults.

**8\. Test Time Augmentations \- Tested, but Not Helpful for Small Objects**

What we tried: We used 3-view TTA (original, horizontally flipped, and 2× upscaled images) and merged the predictions.

What we found: TTA gave only a small trade-off on the natural test set, but significantly increased missed detections on the resolution-degraded set.

Why: With very small objects, even small differences in predicted box locations across views can prevent the boxes from being merged correctly, turning some true detections into misses.

Would revisit if: We used a more scale-aware merging strategy designed specifically for small objects.

**9\. head/helmet Class Imbalance**

Options: Keep the 3:1 helmet-to-head split, oversample head, or use class-weighted loss.

Chose: We trained both arms on the data as-is to keep the comparison fair and consistent.

If we had more time: We would try class-weighted loss or head oversampling and check whether the minority-class weakness improves.

