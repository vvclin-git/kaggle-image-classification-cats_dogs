"""Training entry helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch
from sklearn.model_selection import StratifiedKFold

from .dataset import ds_test_split
from .predict import infer_classifier
from .training import CVTrainer
from .training import Trainer
from .utils import profile_dataloader_grid_search


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


class FinalModelWorkflow:
    """
    Deployment-oriented workflow:
      1) dataloader profiling
      2) cv training
      3) data split
      4) model training
      5) model inference
    """

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = device
        self.results: dict[str, Any] = {}

    def dataloader_profiling(
        self,
        *,
        ds: Any,
        model: torch.nn.Module,
        param_grid: dict[str, list] | list[dict],
        mode: str = "val",
        steps: int = 200,
        warmup: int = 20,
        repeats: int = 1,
        loss_fn: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[str, Any]:
        prof = profile_dataloader_grid_search(
            ds=ds,
            model=model,
            device=self.device,
            param_grid=param_grid,
            loss_fn=loss_fn,
            optimizer=optimizer,
            mode=mode,
            steps=steps,
            warmup=warmup,
            repeats=repeats,
        )
        self.results["dataloader_profiling"] = prof
        return prof

    def cv_training(
        self,
        *,
        model_cls: type[torch.nn.Module],
        model_params: Optional[dict[str, Any]],
        idx_train: list[int],
        y_train: list[int],
        ds: Any,
        ds_aug: Any,
        n_splits: int = 5,
        splitter: Any = None,
        split_groups: Optional[list[Any]] = None,
        epochs: int = 20,
        tr_bs: int = 64,
        val_bs: Optional[int] = None,
        tr_nw: int = 4,
        val_nw: int = 4,
        loss_fn_factory: Any = torch.nn.CrossEntropyLoss,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, list[float]]], Any]:
        if splitter is None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=37)
        hist, oof = run_cv_training(
            model_cls=model_cls,
            model_params=model_params,
            idx_train=idx_train,
            y_train=y_train,
            ds=ds,
            ds_aug=ds_aug,
            splitter=splitter,
            split_groups=split_groups,
            epochs=epochs,
            tr_bs=tr_bs,
            val_bs=val_bs,
            tr_nw=tr_nw,
            val_nw=val_nw,
            loss_fn_factory=loss_fn_factory,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            device=self.device,
        )
        self.results["cv_training"] = {"hist": hist, "oof_pred": oof}
        return hist, oof

    def data_split(
        self,
        *,
        ds: Any,
        test_size: float = 0.2,
        random_state: int = 37,
        stratify: bool = True,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        idx_tr, idx_te, y_tr, y_te = ds_test_split(
            ds=ds,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        self.results["data_split"] = {
            "idx_train": idx_tr,
            "idx_test": idx_te,
            "y_train": y_tr,
            "y_test": y_te,
        }
        return idx_tr, idx_te, y_tr, y_te

    def model_training(
        self,
        *,
        model: torch.nn.Module,
        idx_train: list[int],
        y_train: list[int],
        ds_tr: Any,
        ds_val: Any,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        tr_bs: int,
        val_bs: Optional[int] = None,
        tr_nw: int = 4,
        val_nw: int = 4,
        show_progress: bool = True,
        chk_pt_period: int = 20,
        save_chk_pt: bool = False,
        chk_pt_dir: Any = None,
    ) -> dict[str, list[float]]:
        trainer = Trainer(
            model=model,
            idx_train=idx_train,
            y_train=y_train,
            ds_tr=ds_tr,
            ds_val=ds_val,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
        )
        hist = trainer.train(
            epochs=epochs,
            tr_bs=tr_bs,
            val_bs=val_bs,
            tr_nw=tr_nw,
            val_nw=val_nw,
            show_progress=show_progress,
            chk_pt_period=chk_pt_period,
            save_chk_pt=save_chk_pt,
            chk_pt_dir=chk_pt_dir,
        )
        self.results["model_training"] = {"trainer": trainer, "hist": hist}
        return hist

    def model_inference(
        self,
        *,
        model: torch.nn.Module,
        data: Any,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> dict[str, Any]:
        pred = infer_classifier(
            model=model,
            data=data,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.results["model_inference"] = pred
        return pred
