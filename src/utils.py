"""Utility helpers."""

from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader
import time


def plot_hist(hist_list, figsize=(12, 4), show=True):
    # Normalize input
    if isinstance(hist_list, dict):
        hist_list = [hist_list]

    if not hist_list:
        print("No data to plot.")
        return None, []

    # Keys to plot (list values with data)
    first = hist_list[0]
    keys = [k for k, v in first.items() if isinstance(v, list) and len(v) > 0]
    if not keys:
        print("No data to plot.")
        return None, []

    epochs = range(1, len(first[keys[0]]) + 1)

    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    multi = len(hist_list) > 1

    for ax, key in zip(axes, keys):
        for i, h in enumerate(hist_list):
            series = h.get(key)
            if isinstance(series, list) and len(series) > 0:
                ax.plot(epochs, series, label=f"Fold {i+1}" if multi else key)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.legend()

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_confusion_matrix_and_report(y_true, y_pred, target_names=['not survived', 'survived']):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.show()
    
    print(classification_report(y_true, y_pred, target_names=target_names))

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

def profile_dataloader(
    ds,
    model,
    device,
    batch_size: int,
    num_workers: int,
    loss_fn=None,
    optimizer=None,
    mode: str = "train",          # "train" or "val"
    steps: int = 200,
    warmup: int = 20,
    lr: float = 1e-3,
    prefetch_factor: int | None = None,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    """
    Profiles end-to-end step time split into:
      - load_tf: time waiting for next batch from DataLoader (CPU-side wait)
      - h2d: host-to-device copy time
      - gpu: GPU forward(+backward+step) time measured by CUDA events
      - total: wall time from before next(it) to after GPU sync
      - img_s: throughput (images/sec)

    Notes:
      - load_tf is 'wait time' for batch readiness, not pure transform cost.
      - Requires CUDA for accurate gpu timing with events.
    """
    assert mode in ("train", "val"), "mode must be 'train' or 'val'"

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and (num_workers > 0),
        drop_last=drop_last if mode == "train" else False,
    )
    if prefetch_factor is not None and num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    dl = DataLoader(ds, **dl_kwargs)
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    if mode == "train":
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)        
        model.train()
    else:
        model.eval()

    load_tf_times: list[float] = []
    h2d_times: list[float] = []
    gpu_times: list[float] = []
    total_times: list[float] = []

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    it = iter(dl)

    for step in range(warmup + steps):
        t0 = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        t1 = time.perf_counter()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
            start_evt.record()
            logits = model(x)
            loss = loss_fn(logits, y.long())
            loss.backward()
            optimizer.step()
            end_evt.record()
        else:
            # forward only
            with torch.inference_mode():
                start_evt.record()
                _ = model(x)
                end_evt.record()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        gpu_s = start_evt.elapsed_time(end_evt) / 1000.0

        if step >= warmup:
            load_tf_times.append(t1 - t0)
            h2d_times.append(t2 - t1)
            gpu_times.append(gpu_s)
            total_times.append(t3 - t0)

    avg_load_tf = sum(load_tf_times) / steps
    avg_h2d = sum(h2d_times) / steps
    avg_gpu = sum(gpu_times) / steps
    avg_total = sum(total_times) / steps
    img_s = batch_size / avg_total

    return {
        "mode": mode,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "load_tf": avg_load_tf,
        "h2d": avg_h2d,
        "gpu": avg_gpu,
        "total": avg_total,
        "img_s": img_s,
        "loader_kwargs": dl_kwargs,
    }
