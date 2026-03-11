"""Model helpers."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet,
    ResNet18_Weights,
    efficientnet,
    resnet,
)


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


def _set_trainable_modules(model: nn.Module, train_mods: Optional[Iterable[str]]) -> None:
    if not train_mods:
        return
    for m in model.modules():
        for p in m.parameters():
            p.requires_grad = False
    for name in train_mods:
        module = model.get_submodule(name)
        for p in module.parameters():
            p.requires_grad = True


class _PredictMixin:
    @torch.inference_mode()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.predict_logits(x), dim=1, dtype=torch.float32)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


class ResNet18Clf(_PredictMixin, ResNet):
    def __init__(self, num_classes: int = 2, train_mods: Optional[Iterable[str]] = None) -> None:
        super().__init__(block=resnet.BasicBlock, layers=[2, 2, 2, 2])
        resnet_18_weights = ResNet18_Weights.DEFAULT
        state_dict = resnet_18_weights.get_state_dict(progress=True, check_hash=True)
        self.load_state_dict(state_dict)
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        _set_trainable_modules(self, train_mods)


class EffNetB0Clf(_PredictMixin, efficientnet.EfficientNet):
    def __init__(self, num_classes: int = 2, train_mods: Optional[Iterable[str]] = None) -> None:
        inverted_residual_setting, last_channel = efficientnet._efficientnet_conf(
            "efficientnet_b0",
            width_mult=1.0,
            depth_mult=1.0,
        )
        super().__init__(
            inverted_residual_setting=inverted_residual_setting,
            dropout=0.2,
            last_channel=last_channel,
        )
        weights = EfficientNet_B0_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True, check_hash=True)
        self.load_state_dict(state_dict)
        in_features = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(in_features, num_classes)
        _set_trainable_modules(self, train_mods)


# Backward-compatible aliases for notebook class names.
ResNet18_Clf = ResNet18Clf
EffNet_B0_Clf = EffNetB0Clf
