# Solution Overview: Iterative Data Cleaning & Visual Error Analysis
Summary
My approach focused heavily on a Data-Centric AI workflow. Since the model architecture was restricted to YOLOv8n, I found that changing pre-trained weights yielded little improvement. Instead, I concentrated on refining the dataset using the 3LC Dashboard, aggressive data augmentation, and a custom visual feedback loop.

1. Data-Centric Strategy (3LC)
I utilized the 3LC Dashboard to iteratively clean the dataset. My workflow involved:

Error Identification: I specifically filtered for:

High Confidence / Low IoU: To identify valid objects that were missing from the ground truth (missing labels).

Mid-Range IoU: To find and correct bounding boxes that were misaligned or loosely fitted.

Class Confusion: Manually checking difficult samples where the model was confident but wrong.

Sample Weighting: I applied higher weights to the difficult samples identified during the cleaning process to force the model to focus on these edge cases during training.

2. Training & Hyperparameters
I used the standard YOLOv8n architecture.

Optimizer: I experimented with SGD and AdamW. I found that tuning the SGD momentum (0.937) and weight decay gave stable results.

Augmentations: To prevent overfitting on the small model, I heavily customized the augmentation pipeline in my training script (train.py), including:

Geometric: degrees=15, translate=0.15, scale=0.7, fliplr=0.5.

Color/Lighting: Adjusted hsv_h, hsv_s, and hsv_v to account for different lighting conditions in the field.

Mosaic & Mixup: Enabled mosaic (1.0) and mixup (0.1), along with copy_paste (0.2) to simulate dense weed scenarios.

3. Visual Error Analysis Loop
I developed a custom script, visualize_test_predictions.py, to draw bounding boxes and confidence scores directly onto the test set images.

The Loop: After every training run, I visualized the predictions.

Observation: I looked for patterns where the model consistently failed (e.g., specific weed textures or lighting).

Action: I used these insights to go back to the 3LC Dashboard, find similar examples in the training set, and correct their labels or adjust their weights.

4. Administrative Note: Team Status & Prize Waiver
Important Note to Organizers: I started this competition as part of a different team. However, due to the inactivity of my teammates, I worked independently throughout the challenge. To ensure I could submit my own work, I created this new team entry for my final submission.

I understand this may be a grey area regarding competition rules. Therefore:

I voluntarily waive my right to any monetary prizes.

I am submitting this solely for the purpose of ranking verification and certification of my skills.
