"""Training-related helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


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
