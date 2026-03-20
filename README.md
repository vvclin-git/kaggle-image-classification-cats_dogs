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
- Dataset link: [Microsoft Cats vs Dogs (PetImages) - Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
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

### 6.1 Notebook Workflow
Use the notebooks in order:

1. `notebooks/00_eda.ipynb`
2. `notebooks/01_baseline_train.ipynb`
3. `notebooks/02_improve_train.ipynb`

For deployment-style orchestration inside Python, use `FinalModelWorkflow`:

```python
from sklearn.model_selection import StratifiedKFold
from src import FinalModelWorkflow

wf = FinalModelWorkflow(device=DEVICE)

profile = wf.dataloader_profiling(
    ds=ds_aug,
    model=model_for_profile,
    param_grid={"batch_size": [32, 64], "num_workers": [4, 8]},
)

hist_cv, oof = wf.cv_training(
    model_cls=FinalModelClass,
    model_params={"num_classes": 2},
    idx_train=idx_train,
    y_train=y_train,
    ds=ds,
    ds_aug=ds_aug,
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=37),
)

idx_tr, idx_te, y_tr, y_te = wf.data_split(ds=ds, test_size=0.2, random_state=37, stratify=True)
hist_final = wf.model_training(
    model=final_model,
    idx_train=idx_tr,
    y_train=y_tr,
    ds_tr=ds_train_subset,
    ds_val=ds_val_subset,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
    tr_bs=64,
    val_bs=64,
)
pred = wf.model_inference(model=final_model, data=ds_test_subset, batch_size=128)
```

### 6.2 CLI Workflow
The CLI is split by workflow stage:

1. `profile-dataloader`
2. `cv-train`
3. `split-data`
4. `train-model`
5. `infer-model`
6. `make-submission`

Argument conventions:
1. `--dataset-factory` and `--model-class` are dotted paths.
2. `--model-params-json`, `--optimizer-params-json`, `--param-grid-json` are inline JSON strings.
3. `infer-model` can pass `--dataset-root` into a dataset factory for inference-only datasets.
4. Persist intermediate files between steps (`split.json`, `cv_result.json`, `train_result.json`, checkpoints, inference outputs).

Example commands:

```powershell
# 1) DataLoader profiling
cats-dogs-cli profile-dataloader `
  --dataset-factory my_project.factories:build_dataset_aug `
  --model-class src.model.EffNet_B0_Clf `
  --model-params-json "{\"num_classes\":2,\"train_mods\":[\"features.7\",\"features.8\",\"classifier\"]}" `
  --param-grid-json "{\"batch_size\":[32,64],\"num_workers\":[4,8]}" `
  --mode train --steps 100 --warmup 20 --repeats 1 `
  --out-json outputs/profile.json

# 2) CV training
cats-dogs-cli cv-train `
  --dataset-factory my_project.factories:build_dataset_val `
  --dataset-aug-factory my_project.factories:build_dataset_aug `
  --split-json outputs/split.json `
  --model-class src.model.EffNet_B0_Clf `
  --model-params-json "{\"num_classes\":2,\"train_mods\":[\"features.7\",\"features.8\",\"classifier\"]}" `
  --loss-fn-class torch.nn.CrossEntropyLoss `
  --optimizer-class torch.optim.Adam `
  --optimizer-params-json "{\"lr\":0.001,\"weight_decay\":0.0001}" `
  --n-splits 5 --epochs 5 --tr-bs 64 --val-bs 64 --tr-nw 4 --val-nw 4 `
  --out-json outputs/cv_result.json

# 3) Data split
cats-dogs-cli split-data `
  --dataset-factory my_project.factories:build_dataset_val `
  --test-size 0.2 --random-state 37 --stratify `
  --out-json outputs/split.json

# 4) Final model training
cats-dogs-cli train-model `
  --dataset-factory my_project.factories:build_dataset_val `
  --split-json outputs/split.json `
  --model-class src.model.EffNet_B0_Clf `
  --model-params-json "{\"num_classes\":2,\"train_mods\":[\"features.7\",\"features.8\",\"classifier\"]}" `
  --loss-fn-class torch.nn.CrossEntropyLoss `
  --optimizer-class torch.optim.Adam `
  --optimizer-params-json "{\"lr\":0.001,\"weight_decay\":0.0001}" `
  --epochs 10 --tr-bs 64 --val-bs 64 --tr-nw 4 --val-nw 4 `
  --checkpoint-out outputs/checkpoints/final_effb0.pt `
  --out-json outputs/train_result.json

# 5) Model inference on a labeled dataset split
cats-dogs-cli infer-model `
  --dataset-factory my_project.factories:build_dataset_val `
  --model-class src.model.EffNet_B0_Clf `
  --model-params-json "{\"num_classes\":2,\"train_mods\":[\"features.7\",\"features.8\",\"classifier\"]}" `
  --checkpoint-path outputs/checkpoints/final_effb0.pt `
  --split-json outputs/split.json --use-split test `
  --batch-size 128 --num-workers 4 `
  --out-npz outputs/infer_test.npz --out-json outputs/infer_test.json

# 6) Model inference on an unlabeled folder
cats-dogs-cli infer-model `
  --dataset-factory src.infer_datasets:build_unlabeled_dataset `
  --dataset-root "data/raw/dogs-vs-cats-redux-kernels-edition/test/test" `
  --model-class src.model.EffNet_B0_Clf `
  --model-params-json "{\"num_classes\":2,\"train_mods\":[\"features.7\",\"features.8\",\"classifier\"]}" `
  --checkpoint-path outputs/checkpoints/final_effb0.pt `
  --batch-size 128 --num-workers 4 `
  --out-json outputs/infer_redux_test.json

# 7) Convert inference output to Kaggle submission CSV
cats-dogs-cli make-submission `
  --infer-json outputs/infer_redux_test.json `
  --out-csv outputs/submission.csv
```

Notes:
- `infer-model` accepts checkpoints saved by this project and plain `state_dict` files.
- For unlabeled inference, the output JSON includes `image_paths`, `y_pred`, and `y_proba`.
- `make-submission` writes `id,label` CSV. By default it uses `y_proba[:, 1]`; use `--label-mode pred` to write hard class predictions instead.

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
