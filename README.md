# Fine-Grained Fruit Classification

A deep learning pipeline for fine-grained classification of 20 fruit varieties, featuring an IEL-inspired architecture with Multi-Layer Grad-CAM voting and domain adaptation to robot-captured images.

---

## Overview

Fine-grained visual classification is challenging because classes share high-level structure (e.g., multiple apple varieties look nearly identical). This project tackles a 20-class fruit dataset using two models — **EfficientNet-B3-IEL** and **ResNet50-IEL** — built on top of an Internal Ensemble Learning (IEL) framework. Both models are then fine-tuned on images captured by a robot camera to evaluate cross-domain transfer.

---

## Dataset

The dataset contains 20 fruit classes spread across four fruit groups:

| Group | Classes |
|---|---|
| Apple | `apple_ambrosia`, `apple_bravo`, `apple_grannysmith`, `apple_jazz`, `apple_kanzi`, `apple_mi`, `apple_pinklady`, `apple_royalgala` |
| Banana | `banana_cavendish`, `banana_ladyfinger`, `banana_redtipped` |
| Pear | `pear_beurre_bosc`, `pear_brown_nashi`, `pear_corella`, `pear_green_nashi`, `pear_william_barlett` |
| Other | `kiwi_gold`, `kiwi_green`, `tomato_gourmet`, `tomato_truss` |

- Each class has at least 50 images (phone-captured).
- A separate **robot image** subset (3 apple classes, ~30–40 images each) is used for domain adaptation evaluation.
- Data split: **60% train / 10% validation / 30% test** (stratified).

---

## Architecture

### Inspiration: Ensemble Learning

<img width="1906" height="1070" alt="Image" src="https://github.com/user-attachments/assets/18a8eb3c-b286-47c8-970a-a87c885c6480" />

Our model architecture is inspired by ensemble machine learning methods. A Random Forest, for example, is built from many individual decision trees arranged in a tree structure — each tree sees the data slightly differently and votes on the final prediction. No single tree is trusted on its own; the strength comes from combining many weak learners into one robust decision.

We apply the same intuition to our deep learning model. EfficientNet-B3 is made up of seven blocks of Mobile Inverted Convolutional (MBConv) layers, and each block processes the image at a different level of abstraction — earlier blocks respond to edges and textures, while deeper blocks respond to higher-level shapes and object parts. Rather than relying only on the final block's output, we **tap** multiple blocks and combine their perspectives, just as a Random Forest combines multiple trees.

### Tapping Pipeline

<img width="1904" height="1070" alt="image" src="https://github.com/user-attachments/assets/a8f34320-3ebe-46ba-9cc1-1ff284321198" />

Normally, when an input image passes through EfficientNet-B3, each block produces an output that is passed to the next block — and those intermediate outputs are discarded. In my architecture, I intercept them.

From testing, Blocks 1–5 tend to focus on whole-image context or background regions rather than the fruit itself. I therefore tap **Block 6, Block 7, and the convolutional head** — the three deepest points where features are most discriminative — and feed copies of those feature maps into our IEL head.

These tapped features are used to generate a **Multi-Layer Grad-CAM heatmap**, which asks: *where is the model actually looking?* I then check whether that heatmap overlaps with the fruit foreground (estimated via GrabCut) rather than background clutter like shelves or packaging. If the overlap passes a threshold, the heatmap guides targeted augmentation — either enlarging the focal fruit region or blurring everything outside it.

The key insight is that tapping multiple deep layers produces more reliable and better-localised heatmaps than using the final layer alone, which in turn makes the augmentation more targeted and more effective.


---

### Model Diagram

Both models follow the same IEL-inspired design:

```
Pretrained Backbone (EfficientNet-B3 or ResNet50)
        │
        ├── Intermediate feature maps (via forward hooks on selected blocks)
        │
        ├── Per-layer Projectors  →  Cross-Layer Attention Fusion
        │                                    │
        └── Final backbone features  ──────► Fused representation
                                                    │
                                            Classification Head
```

**Key components:**

- **Multi-Layer Grad-CAM (MLV):** Grad-CAM heatmaps are generated from multiple backbone layers. Layer weights are computed from foreground overlap scores, producing a voted heatmap focused on the fruit region.
- **Cross-Layer Refinement:** Features from tapped backbone blocks are projected to a common dimension and fused via cross-attention, combining low-level texture with high-level semantics.
- **GrabCut Foreground Estimation:** Ensures Grad-CAM augmentations target the fruit rather than the background.
- **Grad-CAM-Guided Augmentation:** In Stage 2, training images receive targeted local augmentations (brightness, contrast, saturation) applied specifically to the high-activation fruit region identified in Stage 1 heatmaps.
- **Label Smoothing CE + IEL Loss:** Training combines the final classifier loss with an auxiliary loss from intermediate feature predictions, weighted by `λ_assist = 0.5`.
- **Weighted Random Sampling:** Addresses class imbalance by oversampling underrepresented classes.

---

## Training Pipeline

Training proceeds in two stages:

### Stage 1 — Baseline Feature Learning
- Backbone partially frozen; only top blocks + IEL head trained.
- Optimizer: AdamW (`lr=2e-4`, `weight_decay=5e-3`)
- Scheduler: CosineAnnealingLR
- Epochs: up to 30 (early stopping, patience=7)
- Goal: learn discriminative features and generate reliable Grad-CAM heatmaps.

### Stage 2 — Grad-CAM-Guided Refinement
- More backbone blocks unfrozen.
- Training data augmented with Grad-CAM-guided local transformations from the Stage 1 heatmap cache.
- Optimizer: AdamW (`lr=2e-3`, `weight_decay=1e-4`)
- Goal: improve focus on fine-grained discriminative regions.

### Phase 3 — Robot Domain Fine-Tuning
- Both Stage 2 models fine-tuned for 15 epochs on robot-captured apple images (~25 train / 5–6 val per class).

---

## Results

### 20-Class Test Set (phone images, 426 test samples)

| Metric | EfficientNet-IEL | ResNet50-IEL |
|---|---|---|
| Test Accuracy | **0.9155** | 0.8592 |
| Test Loss | **0.9596** | 1.0583 |
| Best Val Loss | **0.8904** | 1.0641 |
| Macro Precision | **0.9235** | 0.8727 |
| Macro Recall | **0.9133** | 0.8609 |
| Macro F1-Score | **0.9155** | 0.8626 |

EfficientNet-IEL leads on 18 out of 20 classes. The hardest classes for both models are visually ambiguous apple varieties (`apple_ambrosia`, `apple_kanzi`, `banana_ladyfinger`).

### Robot Image Evaluation (3 apple classes, 24 test images)

| | EfficientNet-IEL | ResNet50-IEL |
|---|---|---|
| **Pre-finetune accuracy** | 0.4583 | 0.1250 |
| **Post-finetune accuracy** | **1.0000** | 0.8333 |

EfficientNet-IEL achieves perfect accuracy after fine-tuning on all three robot apple classes. ResNet50-IEL plateaus at 83.3%, struggling to separate `apple_pinklady` and `apple_royalgala`.

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
Pillow
```

Install via:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow
```

---

## Usage

The notebook is designed to run in **Google Colab** with data stored on Google Drive.

1. Mount Google Drive and set `data_dir` and `PROJECT_DIR` to your dataset and project paths.
2. Run cells sequentially — each section is self-contained with markdown explanations.
3. Checkpoints are saved to `PROJECT_DIR/checkpoints_efficientnet_iel/` and `PROJECT_DIR/checkpoints_resnet/`.
4. To skip retraining, load saved checkpoints directly before the evaluation cells.

---

## Key Design Decisions

- **No color augmentation** — fine-grained fruit classification is sensitive to color and shape; MixUp, CutOut, and color jitter were excluded as they alter discriminative fruit appearance.
- **EfficientNet-B3 over larger models** — compound scaling provides richer features per parameter (10.7M) compared to ResNet50 (23.5M), with better generalisation to robot images.
- **Single unfreeze block in Stage 1** — unfreezing more blocks caused attention to drift to backgrounds; keeping only the top block focused Grad-CAM on the fruit.

---

## Authors

Vinh Nguyen (Paul) Ha - This project is under a major project in COMP8430 - Advanced Computer Vision and Action at Macquarie University.
