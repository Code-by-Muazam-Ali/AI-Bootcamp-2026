# Visual Phishing Detection from Website Screenshots

This notebook approaches phishing-website detection as a **computer vision problem**: instead of parsing URLs or page text, it classifies raw website **screenshots** as *Genuine* or *Phishing* using a CNN trained in PyTorch.

## 1. What the Notebook Does

| Stage | Description |
|---|---|
| Environment setup | Loads PyTorch/torchvision, scikit-learn, PIL, seaborn; detects CPU/GPU. |
| Dataset discovery | Scans `Dataset/genuine_site_0` and `Dataset/phishing_site_1`, builds a labeled DataFrame. |
| Data quality checks | Validates every image opens correctly, computes SHA-256 (exact duplicates) and a perceptual hash (near-duplicates). |
| Visualization | Sample grids of genuine vs. phishing screenshots, class-distribution bar chart. |
| Train/val/test split | Stratified 70/15/15 split on the label column. |
| Preprocessing & augmentation | Resize to 224×224, ImageNet normalization; training-only flips/rotation/color-jitter; class weights computed for the loss function. |
| Model | A compact **Custom CNN** (3 conv blocks → global average pool → dropout → linear), ~94K parameters, trained from scratch. |
| Training | Weighted cross-entropy loss, AdamW optimizer, ReduceLROnPlateau scheduler, early stopping (patience = 4) monitored on validation ROC-AUC, max 20 epochs. |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, FPR, FNR, confusion matrix, ROC/PR curves. |
| Explainability | Grad-CAM heatmaps for correct genuine/phishing, false-positive, and false-negative examples. |
| Error analysis | Lists false-positive and false-negative screenshots. |
| Final summary | Consolidated metrics table and a short "cybersecurity analysis" narrative. |

> **Note:** The introduction describes CNNs, transfer-learning models, a Vision Transformer, and ensembles, but only the **Custom CNN** was actually trained and evaluated in this run — the ensemble/voting cells were emptied out ("Removed ensemble logic as per user request"), so the final comparison table contains a single model.

## 2. Dataset

- **Total images:** 1,697 (matches the expected count exactly)
- **Genuine:** 1,147 (67.6%)
- **Phishing:** 550 (32.4%)
- **Imbalance ratio:** ~2.08 genuine images for every phishing image
- **Splits:** Train 1,187 / Validation 255 / Test 255 (stratified, so class ratio is preserved in each split)

### Data quality issues found by the notebook itself
- **322 exact duplicate files** across 74 duplicate groups (out of 1,697 total) — roughly **19% of the dataset is a byte-for-byte copy of another image**.
- **122 near-duplicate groups** detected via perceptual hashing.
- **Cross-split overlap:** the split-verification step reports **33 overlapping images between train/validation, 27 between train/test, and 13 between validation/test**, even though the notebook's stated goal was zero exact overlaps. This is **data leakage** — the same (or duplicate) screenshot can appear in both the training set and the test set.

**Impact:** Because the model can be trained on an image that is literally identical or near-identical to one it is later "tested" on, the reported test metrics are **optimistically biased**. The true generalization performance on genuinely unseen websites is likely worse than the numbers below.

## 3. Results (Custom CNN, Test Set, n = 255)

| Metric | Value |
|---|---|
| Accuracy | 0.651 (65.1%) |
| Precision (Phishing) | 0.477 (47.7%) |
| Recall (Phishing) | 0.890 (89.0%) |
| F1-score | 0.621 |
| ROC-AUC | 0.772 |
| PR-AUC | 0.529 |
| Specificity (Genuine correctly kept) | 0.538 |
| False Positive Rate | 0.462 |
| False Negative Rate | 0.110 |
| False Positives (genuine flagged as phishing) | 80 |
| False Negatives (phishing missed) | 9 |

Training converged quickly and **early-stopped at epoch 7 of 20** once validation ROC-AUC stopped improving (it hovered around 0.65–0.68 for most of training).

## 4. Why These Results Are Low (and Why Recall Is High)

**Accuracy (65%) and Precision (48%) are low because:**
- **Class imbalance + weighted loss pushed the model toward over-predicting "Phishing."** The loss function up-weights the minority (phishing) class by ~1.54× and down-weights genuine by ~0.74×, which trades away precision for recall — the model calls many genuine sites "phishing" (80 false positives out of 173 genuine test images, a 46% false-positive rate).
- **Data leakage inflates what would otherwise be an even lower score.** With near-duplicate images crossing the train/val/test boundary, some "test" performance is really memorization, not generalization to new sites.
- **The Custom CNN is small and trained from scratch (no pretraining).** With ~94K parameters and only ~1,187 training images (many of them duplicates, so effectively fewer unique examples), the network has limited capacity to learn the subtle visual cues that separate real and spoofed pages, especially compared to a pretrained model with ImageNet features.
- **Aggressive but generic augmentation** (small rotations, color jitter, affine shear) helps a little with overfitting but doesn't compensate for the small, imbalanced, and duplicate-heavy dataset.
- **No transfer learning was actually evaluated.** Pretrained backbones (ResNet, EfficientNet, ViT, etc.) that the intro mentions were never trained in this run, so there's no stronger baseline to compare against or ensemble with.

**Recall (89%) and ROC-AUC (0.77) are comparatively high because:**
- The weighted loss explicitly optimizes for **not missing phishing pages** — which is the security-relevant priority — so the model rarely lets a phishing screenshot through (only 9 false negatives out of 82 phishing test images, an 11% miss rate).
- ROC-AUC, unlike accuracy, is threshold-independent and less sensitive to class imbalance, so it reflects a genuinely reasonable *ranking* ability (the model does put most phishing images at higher predicted probability than most genuine ones) even though the default 0.5 threshold produces many false alarms.

## 5. Real-World Impact of These Numbers

- **High recall / low false-negative rate (11%) is good news for security posture** — the model is unlikely to wave through a phishing page as a false negative, which is the more dangerous failure mode (a user is deceived and may lose credentials or money).
- **Low precision / high false-positive rate (46%) is a serious usability problem.** If deployed as-is (e.g., blocking or warning on flagged pages), nearly half of *legitimate* websites shown to the model would be incorrectly flagged as phishing. In production this would cause:
  - Alert fatigue for security analysts reviewing flagged sites.
  - Broken user experience if legitimate sites are auto-blocked.
  - Erosion of trust in the detector, leading operators to raise the decision threshold or ignore alerts — which would in turn increase false negatives.
- **Because of duplicate images and cross-split leakage, even these numbers should be treated as an upper bound**, not a reliable estimate of how the model would perform on brand-new, previously unseen websites in the wild.
- **The notebook only reports one model**, so there is no evidence yet on whether a pretrained CNN, a Vision Transformer, or an ensemble would meaningfully improve the precision/recall trade-off — the introduction's stated comparison goal was not completed in this run.

## 6. Recommendations for Improving the Results

1. **Deduplicate the dataset before splitting** (drop or explicitly group exact and near-duplicates by hash) so that train/val/test partitions share no visually similar images.
2. **Re-run the split verification** after deduplication and confirm zero exact and near-duplicate overlap across partitions.
3. **Train the pretrained/transfer-learning and ViT models** referenced in the introduction, and restore the soft-voting/weighted ensemble logic that was removed, to get a genuine model comparison.
4. **Tune the decision threshold** (not just rely on 0.5) using the precision-recall curve to find a better operating point for the false-positive/false-negative trade-off appropriate for the deployment context.
5. **Collect more phishing examples** or use stronger class-balancing (e.g., oversampling, focal loss) instead of relying solely on class-weighted cross-entropy.
6. **Use Grad-CAM findings** (already generated in the notebook) to check whether the model is focusing on meaningful visual regions (logos, login forms, layout) rather than incidental artifacts, which would explain some of the false positives.

## 7. Repository / Output Structure

The notebook writes all artifacts under `outputs/`:
```
outputs/
├── models/        # saved checkpoints
├── plots/         # class distribution, ROC/PR curves, training curves
├── gradcam/        # Grad-CAM heatmaps (correct/FP/FN cases)
├── errors/         # misclassified examples
├── predictions/     # per-image prediction CSVs
├── logs/          # training logs
├── reports/        # text reports
└── tables/         # metric comparison CSVs
```

## 8. How to Run

1. Place the dataset under `Dataset/genuine_site_0/` and `Dataset/phishing_site_1/` (or mount Google Drive if running in Colab).
2. Run all cells top to bottom. The notebook auto-creates the `outputs/` folder structure.
3. Runs on CPU or GPU automatically (this run executed on **CPU only**, which is part of why training was capped at 20 epochs with early stopping rather than a longer schedule).

