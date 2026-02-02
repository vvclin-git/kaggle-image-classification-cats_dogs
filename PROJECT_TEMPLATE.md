# Kaggle ML Project Template (Placeholder)

This file captures the current project structure and conventions so you can
replicate similar projects elsewhere. Replace placeholders as needed.

## Project layout
```
<project-name>/
  data/
    raw/                      # Raw data files (format-agnostic)
    processed/                # Cleaned/feature data (optional)
  docs/
    figures/                  # Plots and exports
  notebooks/
    00_eda.ipynb
    01_baseline_train.ipynb
    02_improve_train.ipynb
    best_params.json
    ordinal_mappings.json
    selected_features_log1p.json
    selected_features_mae.json
  outputs/
    models/                   # Saved models (ignored by git)
    params/                   # HPO params json (tracked)
    submissions/              # Kaggle submissions (tracked)
  src/
    config.py                 # Paths, seed, device selection
    dataset.py                # Dataset loading/processing (empty placeholder)
    model.py                  # Model definitions (empty placeholder)
    train.py                  # Training pipeline (empty placeholder)
    predict.py                # Prediction/inference (empty placeholder)
    utils.py                  # Helpers/metrics/saving (empty placeholder)
  .gitignore
  pyproject.toml
  uv.lock
  README.md
```

## Configuration conventions (`src/config.py`)
```
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
PARAMS_PATH = OUTPUT_DIR / "params"

SEED = 37

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
```

## Dependencies (`pyproject.toml`)
- Python >= 3.11
- Core deps: `jupyter`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`,
  `tqdm`, `torch`, `torchvision`, `xgboost`
- Dev deps: `ipykernel`
- `uv.lock` is present (use `uv` if you want reproducible installs)

## PyTorch pinning (version + wheel)
Current versions resolved in this project:
- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- Index URL: `https://download.pytorch.org/whl/cu128`

Wheel naming uses the Python ABI tag and platform tag. Examples:
- Windows + Python 3.12: `torch-2.9.1+cu128-cp312-cp312-win_amd64.whl`
- Windows + Python 3.12: `torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl`

If you generate a fresh `pyproject.toml`, you can pin like this:
```
[project]
dependencies = [
  "torch==2.9.1+cu128",
  "torchvision==0.24.1+cu128",
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

## Git ignore behavior
- Ignores: `data/raw`, `data/processed`, `outputs/models`, `outputs/logs`
- Tracks: `outputs/submissions`, `outputs/params`

## Suggested workflow (fill in `src/`)
1) `notebooks/00_eda.ipynb` for EDA and initial feature ideas.
2) `notebooks/01_baseline_train.ipynb` for quick baselines.
3) `notebooks/02_improve_train.ipynb` for feature engineering and tuning.
4) Move stable logic into `src/`:
   - `dataset.py`: read data, split, preprocess, feature engineering.
   - `model.py`: model definitions and factory functions.
   - `train.py`: training loop, CV, metric computation, save artifacts.
   - `predict.py`: load model, create submission CSV.
   - `utils.py`: shared helpers, metrics, logging, serialization.

## Replication checklist
- Copy the directory layout above.
- Update `pyproject.toml` name/description if needed.
- Place your data files into `data/raw/` (any format).
- Implement `src/` modules (they are empty placeholders in this template).
- Keep outputs in `outputs/`, with submissions tracked and models/logs ignored.
