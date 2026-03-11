"""Command-line entry points for separated training/inference workflows."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from .config import get_device
from .train import FinalModelWorkflow
from .training import save_checkpoint


def _resolve_dotted(path: str) -> Any:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _parse_json(s: str | None, default: Any) -> Any:
    if s is None:
        return default
    return json.loads(s)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _load_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_workflow(device: str | None) -> FinalModelWorkflow:
    return FinalModelWorkflow(device=device or get_device())


def cmd_split_data(args: argparse.Namespace) -> None:
    dataset_factory = _resolve_dotted(args.dataset_factory)
    ds = dataset_factory()
    wf = _build_workflow(args.device)
    idx_tr, idx_te, y_tr, y_te = wf.data_split(
        ds=ds,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=args.stratify,
    )
    _save_json(
        args.out_json,
        {"idx_train": idx_tr, "idx_test": idx_te, "y_train": y_tr, "y_test": y_te},
    )
    print(f"saved split -> {args.out_json}")


def cmd_profile_dataloader(args: argparse.Namespace) -> None:
    dataset_factory = _resolve_dotted(args.dataset_factory)
    model_cls = _resolve_dotted(args.model_class)
    model_params = _parse_json(args.model_params_json, {})
    param_grid = _parse_json(args.param_grid_json, {"batch_size": [64], "num_workers": [4]})

    ds = dataset_factory()
    model = model_cls(**model_params)
    wf = _build_workflow(args.device)
    prof = wf.dataloader_profiling(
        ds=ds,
        model=model,
        param_grid=param_grid,
        mode=args.mode,
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _save_json(args.out_json, prof)
    print(f"saved profile -> {args.out_json}")


def cmd_cv_train(args: argparse.Namespace) -> None:
    dataset_factory = _resolve_dotted(args.dataset_factory)
    dataset_aug_factory = _resolve_dotted(args.dataset_aug_factory) if args.dataset_aug_factory else None
    model_cls = _resolve_dotted(args.model_class)
    loss_fn_cls = _resolve_dotted(args.loss_fn_class)
    optimizer_cls = _resolve_dotted(args.optimizer_class)
    model_params = _parse_json(args.model_params_json, {})
    optimizer_params = _parse_json(args.optimizer_params_json, {})

    split = _load_split(args.split_json)
    idx_train = split["idx_train"]
    y_train = split["y_train"]

    ds = dataset_factory()
    ds_aug = dataset_aug_factory() if dataset_aug_factory else ds

    wf = _build_workflow(args.device)
    hist, oof = wf.cv_training(
        model_cls=model_cls,
        model_params=model_params,
        idx_train=idx_train,
        y_train=y_train,
        ds=ds,
        ds_aug=ds_aug,
        n_splits=args.n_splits,
        epochs=args.epochs,
        tr_bs=args.tr_bs,
        val_bs=args.val_bs,
        tr_nw=args.tr_nw,
        val_nw=args.val_nw,
        loss_fn_factory=loss_fn_cls,
        optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params,
    )
    _save_json(args.out_json, {"hist": hist, "oof_pred": oof})
    print(f"saved cv result -> {args.out_json}")


def cmd_train_model(args: argparse.Namespace) -> None:
    dataset_factory = _resolve_dotted(args.dataset_factory)
    model_cls = _resolve_dotted(args.model_class)
    loss_fn_cls = _resolve_dotted(args.loss_fn_class)
    optimizer_cls = _resolve_dotted(args.optimizer_class)
    model_params = _parse_json(args.model_params_json, {})
    optimizer_params = _parse_json(args.optimizer_params_json, {})
    split = _load_split(args.split_json)

    ds = dataset_factory()
    idx_train = split["idx_train"]
    idx_val = split["idx_test"]
    y_train = split["y_train"]

    ds_tr = Subset(ds, idx_train)
    ds_val = Subset(ds, idx_val)
    model = model_cls(**model_params)
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    loss_fn = loss_fn_cls()

    wf = _build_workflow(args.device)
    hist = wf.model_training(
        model=model,
        idx_train=idx_train,
        y_train=y_train,
        ds_tr=ds_tr,
        ds_val=ds_val,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        tr_bs=args.tr_bs,
        val_bs=args.val_bs,
        tr_nw=args.tr_nw,
        val_nw=args.val_nw,
        show_progress=not args.no_progress,
        chk_pt_period=args.chk_pt_period,
        save_chk_pt=False,
    )

    if args.checkpoint_out:
        save_checkpoint(
            ckpt_path=args.checkpoint_out,
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            hist=hist,
            extra={"tr_bs": args.tr_bs, "val_bs": args.val_bs},
        )
        print(f"saved checkpoint -> {args.checkpoint_out}")

    _save_json(args.out_json, {"hist": hist})
    print(f"saved train result -> {args.out_json}")


def cmd_infer_model(args: argparse.Namespace) -> None:
    dataset_factory = _resolve_dotted(args.dataset_factory)
    model_cls = _resolve_dotted(args.model_class)
    model_params = _parse_json(args.model_params_json, {})

    ds = dataset_factory()
    data = ds
    if args.split_json and args.use_split != "all":
        split = _load_split(args.split_json)
        idx = split["idx_test"] if args.use_split == "test" else split["idx_train"]
        data = Subset(ds, idx)

    model = model_cls(**model_params)
    if args.checkpoint_path:
        payload = torch.load(args.checkpoint_path, map_location=args.device or get_device())
        model.load_state_dict(payload["model_state_dict"])

    wf = _build_workflow(args.device)
    pred = wf.model_inference(
        model=model,
        data=data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.out_npz:
        p = Path(args.out_npz)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **pred)
        print(f"saved inference npz -> {args.out_npz}")
    if args.out_json:
        _save_json(args.out_json, pred)
        print(f"saved inference json -> {args.out_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cats-dogs-cli")
    parser.add_argument("--device", default=None, help="cpu/cuda; default uses src.get_device()")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split-data", help="Workflow 3: data split")
    p_split.add_argument("--dataset-factory", required=True, help="dotted callable path")
    p_split.add_argument("--test-size", type=float, default=0.2)
    p_split.add_argument("--random-state", type=int, default=37)
    p_split.add_argument("--stratify", action="store_true", default=True)
    p_split.add_argument("--out-json", required=True)
    p_split.set_defaults(func=cmd_split_data)

    p_prof = sub.add_parser("profile-dataloader", help="Workflow 1: dataloader profiling")
    p_prof.add_argument("--dataset-factory", required=True, help="dotted callable path")
    p_prof.add_argument("--model-class", required=True, help="dotted class path")
    p_prof.add_argument("--model-params-json", default=None, help="inline json object")
    p_prof.add_argument("--param-grid-json", default=None, help="inline json object/list")
    p_prof.add_argument("--mode", choices=["train", "val"], default="val")
    p_prof.add_argument("--steps", type=int, default=200)
    p_prof.add_argument("--warmup", type=int, default=20)
    p_prof.add_argument("--repeats", type=int, default=1)
    p_prof.add_argument("--out-json", required=True)
    p_prof.set_defaults(func=cmd_profile_dataloader)

    p_cv = sub.add_parser("cv-train", help="Workflow 2: cv training")
    p_cv.add_argument("--dataset-factory", required=True, help="dotted callable path")
    p_cv.add_argument("--dataset-aug-factory", default=None, help="dotted callable path")
    p_cv.add_argument("--split-json", required=True)
    p_cv.add_argument("--model-class", required=True, help="dotted class path")
    p_cv.add_argument("--model-params-json", default=None, help="inline json object")
    p_cv.add_argument("--loss-fn-class", default="torch.nn.CrossEntropyLoss")
    p_cv.add_argument("--optimizer-class", default="torch.optim.Adam")
    p_cv.add_argument("--optimizer-params-json", default=None, help="inline json object")
    p_cv.add_argument("--n-splits", type=int, default=5)
    p_cv.add_argument("--epochs", type=int, default=20)
    p_cv.add_argument("--tr-bs", type=int, default=64)
    p_cv.add_argument("--val-bs", type=int, default=64)
    p_cv.add_argument("--tr-nw", type=int, default=4)
    p_cv.add_argument("--val-nw", type=int, default=4)
    p_cv.add_argument("--out-json", required=True)
    p_cv.set_defaults(func=cmd_cv_train)

    p_tr = sub.add_parser("train-model", help="Workflow 4: final model training")
    p_tr.add_argument("--dataset-factory", required=True, help="dotted callable path")
    p_tr.add_argument("--split-json", required=True)
    p_tr.add_argument("--model-class", required=True, help="dotted class path")
    p_tr.add_argument("--model-params-json", default=None, help="inline json object")
    p_tr.add_argument("--loss-fn-class", default="torch.nn.CrossEntropyLoss")
    p_tr.add_argument("--optimizer-class", default="torch.optim.Adam")
    p_tr.add_argument("--optimizer-params-json", default=None, help="inline json object")
    p_tr.add_argument("--epochs", type=int, required=True)
    p_tr.add_argument("--tr-bs", type=int, required=True)
    p_tr.add_argument("--val-bs", type=int, default=64)
    p_tr.add_argument("--tr-nw", type=int, default=4)
    p_tr.add_argument("--val-nw", type=int, default=4)
    p_tr.add_argument("--chk-pt-period", type=int, default=20)
    p_tr.add_argument("--checkpoint-out", default=None)
    p_tr.add_argument("--no-progress", action="store_true")
    p_tr.add_argument("--out-json", required=True)
    p_tr.set_defaults(func=cmd_train_model)

    p_inf = sub.add_parser("infer-model", help="Workflow 5: model inference")
    p_inf.add_argument("--dataset-factory", required=True, help="dotted callable path")
    p_inf.add_argument("--model-class", required=True, help="dotted class path")
    p_inf.add_argument("--model-params-json", default=None, help="inline json object")
    p_inf.add_argument("--checkpoint-path", default=None)
    p_inf.add_argument("--split-json", default=None)
    p_inf.add_argument("--use-split", choices=["train", "test", "all"], default="test")
    p_inf.add_argument("--batch-size", type=int, default=128)
    p_inf.add_argument("--num-workers", type=int, default=4)
    p_inf.add_argument("--out-npz", default=None)
    p_inf.add_argument("--out-json", default=None)
    p_inf.set_defaults(func=cmd_infer_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

