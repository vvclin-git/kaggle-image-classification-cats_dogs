"""Inference helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def _as_dataloader(
    data: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def infer_classifier(
    model: torch.nn.Module,
    data: Any,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, np.ndarray]:
    """
    Run batched inference for a classifier model.

    Returns:
      {
        "y_pred": np.ndarray[int],
        "y_proba": np.ndarray[float],
        "y_true": np.ndarray[int] (only if labels are present in dataset/batches),
      }
    """
    model.eval()
    dl = _as_dataloader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    all_preds: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in dl:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                all_true.append(y.detach().cpu().numpy().astype(int))
            else:
                x = batch

            x = x.to(device, non_blocking=True)
            if hasattr(model, "predict_proba"):
                proba_t = model.predict_proba(x)
            else:
                logits = model(x)
                proba_t = torch.softmax(logits, dim=1, dtype=torch.float32)
            pred_t = torch.argmax(proba_t, dim=1)

            all_proba.append(proba_t.detach().cpu().numpy())
            all_preds.append(pred_t.detach().cpu().numpy().astype(int))

    out: dict[str, np.ndarray] = {
        "y_pred": np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=int),
        "y_proba": np.concatenate(all_proba, axis=0) if all_proba else np.array([]),
    }
    if all_true:
        out["y_true"] = np.concatenate(all_true, axis=0)
    return out
