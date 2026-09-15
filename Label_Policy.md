**The data**

* Hard Hat Workers v10 ("raw\_AllClasses"), 7,035 images, COCO format, no augmentation applied — just auto-orientation fixes.  
* Three real classes: **`head`, `helmet`, `person`**. A fourth category, **`"Workers"`**, exists in the file but has zero annotations — an unused Roboflow placeholder, safely ignored.

**Head and helmet are two states, not two objects**

* We assumed at first that head and helmet might be two separate boxes on the same person — a head box, plus a hard-hat box on top of it.  
* We checked: across 4,612 head boxes, only 258 overlap a helmet box at all, and the overlap is tiny (median IoU 0.06). Only 3 pairs look like a genuine double-label.  
* So each headgear region gets one label: head (bare \= violation) or helmet (covered \= compliant). No matching step needed — we just classify each detected region as one or the other.

**Why we dropped person class**

* Only \~2.2–2.4% of labels are person, consistently across every split — a deliberate annotator choice, not noise. Too sparse to train on.  
* The few person boxes we do have are mostly close/medium-range shots (median 50% of frame height) — not the small, distant people this model needs to recognize.  
* We use the head class as our stand-in for "a person," and check "is this in the restricted area" with a simple zone drawn on the camera view — no person detector needed for that part.  
* Note: We are accepting that if the system fully misses someone's head (too small, hidden), there's no backup signal that a person was there. It just skips them instead of flagging "not sure."

**head/helmet Class Imbalance**

* helmet and head are imbalanced at roughly 3:1 — 13,919 vs. 4,612 boxes (\~75%/25%). We trained on this distribution as-is for both arms, without oversampling or class weighting.

**Hard cases**

* Near-duplicate boxes (same class, heavy overlap): 1 found, out of 18,531 annotations. Broken/degenerate boxes: 0 found. We identified this but did not remove it because one stray box will not provide any meaningful effect on training. Thus we left the data untouched. Flagging it here for EDA purpose.  
* 12 of 450 person boxes had no head/helmet inside. We looked at these manually. Some were people facing away where the head was genuinely not visible, some look like small/blurry background people for which the annotator likely skipped. We treat this as a sign that labeling gets less reliable for small, far-away people.

**Major takeaway: this data doesn't look like the real cameras** 

* A typical head here takes up to 10% of the image height(approximately). On the real cameras (1080p, people far away as stated in the problem statement), a head would be more like 1.4–2.8% of the image height.   
* Only 0.3% of head boxes in this dataset match that real-world scale.

**What we didn't do**

* No manual re-labeling of individual boxes, and no removal of the 1 identified duplicate (negligible effect, not worth the extra step) and no attempt to add back annotators' missed heads/people.  
* Both models trained on the identical, unmodified label set (only person dropped, and merged into one class for Arm 2's stage 1), so the comparison between them stays fair.


