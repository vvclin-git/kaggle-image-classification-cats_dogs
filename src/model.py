"""Model helpers."""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def summarize_top_sections(model: nn.Module) -> None:
    print(f"{'name':20s} {'type':25s} {'total_params':>15s} {'trainable':>15s}")
    print("-" * 80)
    for name, module in model.named_children():
        total, trainable = count_params(module)
        print(f"{name:20s} {module.__class__.__name__:25s} {total:15,d} {trainable:15,d}")


def summarize_children(module: nn.Module) -> None:
    print(f"{'name':20s} {'type':25s} {'total_params':>15s} {'trainable':>15s}")
    print("-" * 80)
    for name, child in module.named_children():
        total, trainable = count_params(child)
        print(f"{name:20s} {child.__class__.__name__:25s} {total:15,d} {trainable:15,d}")
