"""Training entry helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .training import CVTrainer


def run_cv_training(
    *,
    model_cls: type[torch.nn.Module],
    model_params: Optional[dict[str, Any]],
    idx_train: list[int],
    y_train: list[int],
    ds: Any,
    ds_aug: Any,
    splitter: Any,
    split_groups: Optional[list[Any]] = None,
    epochs: int = 20,
    tr_bs: int = 64,
    val_bs: Optional[int] = None,
    tr_nw: int = 4,
    val_nw: int = 4,
    loss_fn_factory: Any = torch.nn.CrossEntropyLoss,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_params: Optional[dict[str, Any]] = None,
    device: str | torch.device = "cpu",
) -> tuple[list[dict[str, list[float]]], Any]:
    """Create a CV trainer and run training in one call."""
    trainer = CVTrainer(
        model_cls=model_cls,
        model_params=model_params,
        idx_train=idx_train,
        y_train=y_train,
        ds=ds,
        ds_aug=ds_aug,
        splitter=splitter,
        split_groups=split_groups,
        loss_fn_factory=loss_fn_factory,
        optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params,
        device=device,
    )
    return trainer.train(
        epochs=epochs,
        tr_bs=tr_bs,
        val_bs=val_bs,
        tr_nw=tr_nw,
        val_nw=val_nw,
    )
