# Cross-View Pseudo-Label Co-Training with DINOv2 and CLIP on CIFAR-10

## 1. Overview

This project explores semi-supervised image classification using two pretrained vision encoders: DINOv2 and CLIP. It implements a simplified cross-view pseudo-label co-training approach inspired by Meta Co-Training (MCT).

The objective is to investigate whether knowledge transferred between complementary visual representations can improve classification performance when labelled data are limited.

## 2. Dataset

The experiment uses CIFAR-10, comprising 60,000 images across 10 classes.

- Training set: 50,000 images.
- Test set: 10,000 images.
- Development subset: 45,000 images.
- Validation subset: 5,000 images.
- Labelled co-training subset: 9,000 images (20%).
- Unlabelled subset: 36,000 images (80%).

A stratified split with random seed 42 was used.

## 3. Methodology

### Feature extraction

Two pretrained encoders were used as frozen feature extractors:

- **DINOv2 ViT-L/14:** 1,024-dimensional representations.
- **CLIP ViT-B/16:** 768-dimensional representations.

Images were resized and centre-cropped to 224 × 224 pixels before model-specific normalisation.

### Classification

Separate MLP classifiers with three hidden layers of 1,024 units were trained using AdamW and cross-entropy loss.

Two supervised baselines were established for each encoder:
- Full-label baseline using 45,000 labelled samples.
- Matched-label baseline using 9,000 labelled samples.

### Cross-view co-training

Both classifiers underwent five epochs of supervised warm-up before reciprocal pseudo-label learning.

Each classifier generated pseudo-labels for unlabelled samples. Predictions with confidence ≥ 0.80 were accepted and transferred to the other classifier.

The combined objective was:

\[
\mathcal{L}=\mathcal{L}_{supervised}+0.5\mathcal{L}_{pseudo}
\]

Co-training was performed for 10 epochs with a batch size of 256.

**Note:** This is a simplified cross-view pseudo-label co-training adaptation inspired by MCT. It does not implement the original algorithm's bilevel meta-gradient optimisation.

## 4. Results

Performance was evaluated on the untouched 10,000-image test set.

| Model | Labelled samples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| DINOv2 baseline | 45,000 | 98.93% | 98.93% |
| DINOv2 baseline | 9,000 | 98.81% | 98.81% |
| DINOv2 co-trained | 9,000 | 98.60% | 98.60% |
| CLIP baseline | 45,000 | 93.60% | 93.60% |
| CLIP baseline | 9,000 | 92.71% | 92.71% |
| CLIP co-trained | 9,000 | 94.13% | 94.13% |

### Key findings

- DINOv2 consistently outperformed CLIP across the experiments.
- Co-training improved CLIP accuracy by 1.42 percentage points compared with its matched-label baseline.
- DINOv2 accuracy decreased slightly by 0.21 percentage points following co-training.
- Pseudo-label acceptance reached 98.14% for DINOv2 and 99.37% for CLIP in the final epoch.

These findings suggest that cross-view pseudo-label learning may benefit weaker feature representations more than stronger ones.

## 5. Limitations

- The implementation does not reproduce the original MCT meta-gradient procedure.
- Only one random train/validation split was evaluated.
- Pretrained encoders remained frozen.
- Pseudo-label confidence does not guarantee correctness.
- Results are limited to CIFAR-10 and require further validation.

## 6. Conclusion

Cross-view pseudo-label co-training improved CLIP classification performance under a limited labelled-data setting but did not improve DINOv2. The results highlight the potential value of transferring information between complementary visual representations, while also demonstrating that co-training benefits may depend on the quality of the participating models.