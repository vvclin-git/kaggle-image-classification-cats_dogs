"""Training-related helpers."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import SEED

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


class CVTrainer:
    """K-fold/stratified CV trainer for torch classification models."""

    def __init__(
        self,
        *,
        model_cls: type[torch.nn.Module],
        model_params: Optional[dict[str, Any]],
        idx_train: list[int] | np.ndarray,
        y_train: list[int] | np.ndarray,
        ds: Any,
        ds_aug: Any,
        splitter: Any,
        split_groups: Optional[list[Any] | np.ndarray] = None,
        loss_fn_factory: Callable[[], torch.nn.Module] = torch.nn.CrossEntropyLoss,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict[str, Any]] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model_cls = model_cls
        self.model_params = model_params or {}
        self.idx_train = np.asarray(idx_train)
        self.y_train = np.asarray(y_train)
        self.ds = ds
        self.ds_aug = ds_aug
        self.splitter = splitter
        self.split_groups = split_groups
        self.loss_fn_factory = loss_fn_factory
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params or {}
        self.device = device

        self.hist: list[dict[str, list[float]]] = []
        self.oof_y_pred: Optional[np.ndarray] = None

    def _make_split_iterator(self, show_progress: bool) -> Any:
        if self.split_groups is None:
            split_iter = self.splitter.split(self.idx_train, self.y_train)
        else:
            split_iter = self.splitter.split(
                self.idx_train,
                self.y_train,
                groups=self.split_groups,
            )

        if not show_progress:
            return split_iter

        total = None
        if hasattr(self.splitter, "get_n_splits"):
            try:
                total = self.splitter.get_n_splits()
            except Exception:
                total = None
        return tqdm(split_iter, total=total, desc="CV folds")

    def train(
        self,
        epochs: int,
        tr_bs: int,
        val_bs: Optional[int] = None,
        *,
        tr_nw: int = 4,
        val_nw: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        show_progress: bool = True,
    ) -> tuple[list[dict[str, list[float]]], np.ndarray]:
        if val_bs is None:
            val_bs = tr_bs

        self.hist = []
        self.oof_y_pred = np.zeros(len(self.y_train), dtype=int)
        fold_iter = self._make_split_iterator(show_progress=show_progress)

        for fold, (tr_idx, val_idx) in enumerate(fold_iter):
            fold_hist = {"loss": [], "val_loss": []}

            fold_tr_idx = self.idx_train[tr_idx]
            fold_val_idx = self.idx_train[val_idx]

            fold_tr_dl = DataLoader(
                Subset(self.ds_aug, fold_tr_idx),
                batch_size=tr_bs,
                num_workers=tr_nw,
                shuffle=True,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and (tr_nw > 0),
            )
            fold_val_dl = DataLoader(
                Subset(self.ds, fold_val_idx),
                batch_size=val_bs,
                num_workers=val_nw,
                shuffle=False,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and (val_nw > 0),
            )

            model = self.model_cls(**self.model_params).to(self.device)
            loss_fn = self.loss_fn_factory()
            optim = self.optimizer_cls(model.parameters(), **self.optimizer_params)

            for epoch in range(epochs):
                model.train()
                tr_tot_bs = 0
                total_loss = 0.0
                train_iter = (
                    tqdm(fold_tr_dl, desc=f"fold {fold + 1} train e{epoch + 1}", leave=False)
                    if show_progress
                    else fold_tr_dl
                )
                for x, y in train_iter:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    optim.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = loss_fn(logits, y.long())
                    bs = x.size(0)
                    tr_tot_bs += bs
                    total_loss += float(loss.item()) * bs
                    loss.backward()
                    optim.step()
                fold_hist["loss"].append(total_loss / max(tr_tot_bs, 1))

                model.eval()
                val_tot_bs = 0
                total_val_loss = 0.0
                val_iter = (
                    tqdm(fold_val_dl, desc=f"fold {fold + 1} val e{epoch + 1}", leave=False)
                    if show_progress
                    else fold_val_dl
                )
                with torch.inference_mode():
                    for x, y in val_iter:
                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)
                        logits = model(x)
                        loss = loss_fn(logits, y.long())
                        bs = x.size(0)
                        val_tot_bs += bs
                        total_val_loss += float(loss.item()) * bs
                fold_hist["val_loss"].append(total_val_loss / max(val_tot_bs, 1))

            self.hist.append(fold_hist)

            model.eval()
            fold_y_preds: list[np.ndarray] = []
            pred_iter = (
                tqdm(fold_val_dl, desc=f"fold {fold + 1} oof", leave=False)
                if show_progress
                else fold_val_dl
            )
            with torch.inference_mode():
                for x, _ in pred_iter:
                    x = x.to(self.device, non_blocking=True)
                    if hasattr(model, "predict_proba"):
                        logits = model.predict_proba(x)
                    else:
                        logits = model(x)
                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
                    fold_y_preds.append(preds)

            fold_y_pred = (
                np.concatenate(fold_y_preds, axis=0)
                if fold_y_preds
                else np.array([], dtype=int)
            )
            self.oof_y_pred[val_idx] = fold_y_pred

            del fold_tr_dl, fold_val_dl, model, loss_fn, optim
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.hist, self.oof_y_pred


class Trainer:
    """Single-split trainer (train/val) used by improve notebook."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        idx_train: list[int] | np.ndarray,
        y_train: list[int] | np.ndarray,
        ds_tr: Subset,
        ds_val: Subset,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.idx_train = np.asarray(idx_train)
        self.y_train = np.asarray(y_train)
        self.ds_tr = ds_tr
        self.ds_val = ds_val
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.hist: dict[str, list[float]] = {"loss": [], "val_loss": []}

    def train(
        self,
        epochs: int,
        tr_bs: int,
        val_bs: int | None = None,
        *,
        tr_nw: int = 4,
        val_nw: int = 4,
        show_progress: bool = True,
        chk_pt_period: int = 20,
        save_chk_pt: bool = False,
        chk_pt_dir: Path | None = None,
    ) -> dict[str, list[float]]:
        if val_bs is None:
            val_bs = tr_bs
        if save_chk_pt and chk_pt_dir is None:
            raise ValueError("chk_pt_dir must be provided when save_chk_pt=True")

        tr_dl = DataLoader(
            self.ds_tr,
            tr_bs,
            shuffle=True,
            num_workers=tr_nw,
            persistent_workers=tr_nw > 0,
            pin_memory=True,
        )
        val_dl = DataLoader(
            self.ds_val,
            val_bs,
            shuffle=False,
            num_workers=val_nw,
            persistent_workers=val_nw > 0,
            pin_memory=True,
        )

        epoch_iter = (
            tqdm(range(epochs), desc="epoch", leave=False)
            if show_progress
            else range(epochs)
        )

        for e in epoch_iter:
            if show_progress and hasattr(epoch_iter, "set_description"):
                epoch_iter.set_description(f"epoch {e + 1}/{epochs}")

            self.model.train()
            total_loss = 0.0
            fold_tr_size = 0
            train_iter = (
                tqdm(tr_dl, desc=f"train e{e + 1}", leave=False) if show_progress else tr_dl
            )
            for x, y in train_iter:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = self.loss_fn(logits, y.long())
                bs = x.size(0)
                fold_tr_size += bs
                total_loss += float(loss.item()) * bs
                loss.backward()
                self.optimizer.step()
            train_loss = total_loss / max(fold_tr_size, 1)
            self.hist["loss"].append(train_loss)

            self.model.eval()
            total_val_loss = 0.0
            fold_val_size = 0
            val_iter = (
                tqdm(val_dl, desc=f"val e{e + 1}", leave=False) if show_progress else val_dl
            )
            with torch.inference_mode():
                for x, y in val_iter:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y.long())
                    bs = x.size(0)
                    fold_val_size += bs
                    total_val_loss += float(loss.item()) * bs
            val_loss = total_val_loss / max(fold_val_size, 1)
            self.hist["val_loss"].append(val_loss)

            if show_progress and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

            if (e + 1) % chk_pt_period == 0:
                if show_progress:
                    print(
                        f"epoch: {e + 1}: train loss: {train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
                if save_chk_pt:
                    timestr = time.strftime("%Y%m%d-%H-%M-%S")
                    ckpt_path = f"ckpt_{timestr}_{e + 1}.pt"
                    save_checkpoint(
                        ckpt_path=chk_pt_dir / ckpt_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=e + 1,
                        hist=self.hist,
                        extra={
                            "seed": SEED,
                            "batch_size": tr_bs,
                            "val_batch_size": val_bs,
                        },
                    )

        del tr_dl, val_dl
        return self.hist


def save_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    hist: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    # Save model/optimizer states for exact resume
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "hist": hist,
        "extra": extra or {},
        "pytorch_version": torch.__version__,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    # Load states; returns dict with epoch/hist/extra, etc.
    ckpt_path = Path(ckpt_path)
    payload: dict[str, Any] = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(payload["model_state_dict"], strict=strict)

    if optimizer is not None:
        opt_sd = payload.get("optimizer_state_dict", None)
        if opt_sd is None:
            raise KeyError("Checkpoint has no optimizer_state_dict, cannot resume optimizer.")
        optimizer.load_state_dict(opt_sd)

    return payload


def get_last_checkpoint(ckpt_dir: str | Path, pattern: str = "ckpt_epoch_*.pt") -> Optional[Path]:
    # Find latest checkpoint by epoch number in filename
    ckpt_dir = Path(ckpt_dir)
    cands = sorted(ckpt_dir.glob(pattern))
    if not cands:
        return None

    def parse_epoch(p: Path) -> int:
        s = p.stem  # e.g., ckpt_epoch_120
        try:
            return int(s.split("_")[-1])
        except Exception:
            return -1

    cands = sorted(cands, key=parse_epoch)
    return cands[-1]
