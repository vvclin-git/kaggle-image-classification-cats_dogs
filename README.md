# Kaggle CV Project – Cats vs Dogs Image Classification

## 1. Problem Statement
Predict whether a pet image belongs to the **cat** or **dog** class.

- Task type: Binary image classification
- Evaluation metrics (notebook workflow):
  - Cross-entropy / validation loss for training and cross validation
  - Macro F1 and balanced accuracy for held-out test evaluation

---

## 2. Dataset
- Source: `PetImages` image folder dataset
- Input: Pet images loaded from `ImageFolder`
- Target: Binary class label (`cat` / `dog`)
- Data cleaning:
  - Corrupted image files are detected and excluded
  - Exact duplicate images are detected by file size + SHA256 hash
  - Mislabeled / problematic duplicate groups are added to an ignore list
- Effective dataset size after filtering:
  - Valid samples: **24,968**
  - Excluded samples: **32**
- Dataset split strategy:
  - Train / test split: **50% / 50%**
  - Model selection on the training split via **5-fold stratified cross validation**

---

## 3. Exploratory Data Analysis (EDA)
Key observations from EDA:
- `RGB` is the dominant image format, while a small number of images appear in other modes such as `P`, `L`, `RGBA`, and `CMYK`; these should be converted to `RGB` during preprocessing.
- Image dimensions vary across samples, but most images are roughly within the **300–500 px** range in width and height.
- The class distribution is approximately balanced.
- Duplicate files, corrupted files, and mislabeled error images should be filtered before training.

Relevant plots and examples can be found in:
`notebooks/00_eda.ipynb`

---

## 4. Approach

### 4.1 Baseline
A simple CNN baseline was implemented in PyTorch to establish a performance reference.

- Model: Custom CNN classifier (`Conv-BN-ReLU-MaxPool` stack + global average pooling + linear head)
- Model size:
  - Trainable parameters: **1.08M**
  - Model size: **4.12 MB**
- Goal:
  - Sanity check the data / training pipeline
  - Confirm the model can learn beyond random guessing
  - Establish baseline loss and F1 performance before transfer learning
- Additional work:
  - DataLoader throughput profiling was performed to improve training efficiency
  - A 5-fold CV run with 20 epochs was used to inspect stability and learning behavior

### 4.2 Improved Training
This section focuses on improving performance with transfer learning.

- Candidate models:
  - ResNet18
  - EfficientNet-B0
- Method:
  - Load pretrained torchvision weights
  - Freeze most layers and fine-tune selected late-stage blocks / classifier head
  - Compare candidates with 5-fold stratified CV
- Selection criterion:
  - Validation loss stability
  - Out-of-fold accuracy
  - Accuracy-to-model-size trade-off
- Final choice:
  - **EfficientNet-B0**, because it achieved better CV performance with a substantially smaller trainable footprint than ResNet18

## 5. Results

**Model Comparison / Final Results**

| Stage | Model | Key Result | Notes |
|-------|-------|------------|-------|
| Baseline | Dummy classifier | Random-guess reference | Used for lower-bound comparison |
| Baseline | Custom CNN | Test F1: **0.95** | 1.08M params / 4.12 MB |
| Improve (CV) | ResNet18 | Mean best val loss: **0.0488 ± 0.0081** | OOF accuracy: **0.98** |
| Improve (CV) | EfficientNet-B0 | Mean best val loss: **0.0343 ± 0.0050** | OOF accuracy: **0.99** |
| Final | EfficientNet-B0 | Balanced accuracy: **98.87%** | Macro F1: **98.87%** |

Notebook snapshot date: <2026-03-11>

**Actual Model Performance**

The final EfficientNet-B0 model achieved near-99% precision / recall / F1 on the held-out test split while keeping the model size to about **15 MB**. Compared with the baseline CNN, transfer learning provided a clear performance gain, and compared with ResNet18, EfficientNet-B0 offered the better accuracy-to-size trade-off.

---

## 6. How to Run


---

## 7. Project Structure
```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 00_eda.ipynb
|   |-- 01_baseline_train.ipynb
|   `-- 02_improve_train.ipynb
|-- outputs/
|   |-- checkpoints/
|   |-- models/
|   `-- params/
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- training.py
|   `-- utils.py
`-- README.md
```
---
