"""Utility helpers."""

from __future__ import annotations
import itertools
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
import time
import numpy as np

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


def profile_dataloader_grid_search(
    ds,
    model,
    device,
    param_grid: dict[str, list] | list[dict],
    *,
    loss_fn=None,
    optimizer=None,
    mode: str = "val",
    steps: int = 200,
    warmup: int = 20,
    lr: float = 1e-3,
    repeats: int = 1,
):
    """
    Grid-search DataLoader settings and profile throughput.

    `param_grid` supports two forms:
    - dict[str, list]: cartesian product over all values
    - list[dict]: explicit list of configurations

    Each configuration must include at least:
    - batch_size
    - num_workers

    Returns:
      {
        "best": {...},
        "results": [...],
        "plot_data": {
            "labels": [...],
            "img_s": [...],
            "total": [...],
            "load_tf": [...],
            "h2d": [...],
            "gpu": [...],
        },
      }
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    if isinstance(param_grid, dict):
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    elif isinstance(param_grid, list):
        configs = param_grid
    else:
        raise TypeError("param_grid must be dict[str, list] or list[dict]")

    results = []
    for config in configs:
        if "batch_size" not in config or "num_workers" not in config:
            raise ValueError("each config must include 'batch_size' and 'num_workers'")

        rep_metrics = []
        for _ in range(repeats):
            metrics = profile_dataloader(
                ds=ds,
                model=model,
                device=device,
                batch_size=int(config["batch_size"]),
                num_workers=int(config["num_workers"]),
                loss_fn=loss_fn,
                optimizer=optimizer,
                mode=mode,
                steps=steps,
                warmup=warmup,
                lr=lr,
                prefetch_factor=config.get("prefetch_factor"),
                persistent_workers=config.get("persistent_workers", True),
                pin_memory=config.get("pin_memory", True),
                drop_last=config.get("drop_last", True),
            )
            rep_metrics.append(metrics)

        avg_metrics = {
            "mode": mode,
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "prefetch_factor": config.get("prefetch_factor"),
            "persistent_workers": config.get("persistent_workers", True),
            "pin_memory": config.get("pin_memory", True),
            "drop_last": config.get("drop_last", True),
            "load_tf": sum(m["load_tf"] for m in rep_metrics) / repeats,
            "h2d": sum(m["h2d"] for m in rep_metrics) / repeats,
            "gpu": sum(m["gpu"] for m in rep_metrics) / repeats,
            "total": sum(m["total"] for m in rep_metrics) / repeats,
            "img_s": sum(m["img_s"] for m in rep_metrics) / repeats,
        }
        avg_metrics["config_label"] = (
            f"bs={avg_metrics['batch_size']}, "
            f"nw={avg_metrics['num_workers']}, "
            f"pf={avg_metrics['prefetch_factor']}, "
            f"pw={avg_metrics['persistent_workers']}, "
            f"pin={avg_metrics['pin_memory']}"
        )
        results.append(avg_metrics)

    results = sorted(results, key=lambda x: x["img_s"], reverse=True)
    best = results[0] if results else None

    plot_data = {
        "labels": [r["config_label"] for r in results],
        "img_s": [r["img_s"] for r in results],
        "total": [r["total"] for r in results],
        "load_tf": [r["load_tf"] for r in results],
        "h2d": [r["h2d"] for r in results],
        "gpu": [r["gpu"] for r in results],
    }

    return {
        "best": best,
        "results": results,
        "plot_data": plot_data,
    }


def build_profile_heatmap_data(
    profile_results: dict | list[dict],
    x_param: str,
    y_param: str,
    metrics: list[str] | None = None,
    *,
    agg: str = "mean",
    fill_value=None,
):
    """
    Build heatmap-ready matrices from DataLoader profiling results.

    Args:
      profile_results:
        - output dict from `profile_dataloader_grid_search`, or
        - list of result dicts with config + metric fields.
      x_param, y_param:
        DataLoader config fields for heatmap axes (e.g. "num_workers", "batch_size").
      metrics:
        Metrics for cell color values. Defaults to ["img_s"].
        Supported: "img_s", "gpu", "h2d", "load_tf", "total".
        Alias: "load_ft" -> "load_tf".
      agg:
        How to aggregate multiple runs per same (x, y) cell:
        "mean" (default), "median", "min", "max".
      fill_value:
        Value to place for missing cells.

    Returns:
      {
        "x_param": ...,
        "y_param": ...,
        "x_values": [...],
        "y_values": [...],
        "metrics": [...],
        "matrices": {
            "img_s": [[...], ...],   # shape: [len(y_values)][len(x_values)]
            ...
        }
      }
    """
    if isinstance(profile_results, dict):
        rows = profile_results.get("results")
        if rows is None:
            raise ValueError("profile_results dict must contain a 'results' field")
    elif isinstance(profile_results, list):
        rows = profile_results
    else:
        raise TypeError("profile_results must be dict or list[dict]")

    if not rows:
        return {
            "x_param": x_param,
            "y_param": y_param,
            "x_values": [],
            "y_values": [],
            "metrics": [] if metrics is None else metrics,
            "matrices": {},
        }

    metric_alias = {"load_ft": "load_tf", "throughput": "img_s"}
    metrics = ["img_s"] if metrics is None else metrics
    resolved_metrics = [metric_alias.get(m, m) for m in metrics]
    allowed_metrics = {"img_s", "gpu", "h2d", "load_tf", "total"}
    invalid = [m for m in resolved_metrics if m not in allowed_metrics]
    if invalid:
        raise ValueError(f"unsupported metrics: {invalid}")

    if agg not in {"mean", "median", "min", "max"}:
        raise ValueError("agg must be one of: mean, median, min, max")

    for row in rows:
        if x_param not in row or y_param not in row:
            raise ValueError(f"each row must contain x_param='{x_param}' and y_param='{y_param}'")
        for m in resolved_metrics:
            if m not in row:
                raise ValueError(f"each row must contain metric '{m}'")

    x_values = sorted({row[x_param] for row in rows}, key=lambda v: str(v))
    y_values = sorted({row[y_param] for row in rows}, key=lambda v: str(v))
    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}

    cell_lists = {
        m: [[[] for _ in x_values] for _ in y_values]
        for m in resolved_metrics
    }
    for row in rows:
        xi = x_index[row[x_param]]
        yi = y_index[row[y_param]]
        for m in resolved_metrics:
            cell_lists[m][yi][xi].append(float(row[m]))

    def _reduce(vals: list[float]):
        if not vals:
            return fill_value
        if agg == "mean":
            return sum(vals) / len(vals)
        if agg == "median":
            return statistics.median(vals)
        if agg == "min":
            return min(vals)
        return max(vals)

    matrices = {
        m: [[_reduce(cell) for cell in row] for row in cell_lists[m]]
        for m in resolved_metrics
    }

    return {
        "x_param": x_param,
        "y_param": y_param,
        "x_values": x_values,
        "y_values": y_values,
        "metrics": resolved_metrics,
        "matrices": matrices,
    }

def display_model_info(model):
    """Display model parameter count and size in MB."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Trainable parameters: {param_count}")
    print(f"Model size: {size_mb:.2f} MB")
