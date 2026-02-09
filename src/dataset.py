from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from PIL import Image, UnidentifiedImageError


def get_meta(path: str | Path) -> tuple[str, int, int]:
    with Image.open(path) as im:
        return im.mode, im.size[0], im.size[1]


def is_valid_image(path: str | Path) -> bool:
    try:
        # verify() checks file integrity without fully decoding pixel data
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError, IOError):
        return False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # Hash file content to detect exact duplicates
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dups_size_hash(
    root_dir: Path,
    abs_paths: list[str],
    max_workers: int | None = None,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """
    Returns:
      - dup_groups: {sha256: [rel_paths...]} only groups with len>1
      - dup_files_list: flat list of duplicate rel_paths to drop (keeps 1 per group)
      - keep_files_list: flat list of kept rel_paths (unique representatives)
    """

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 8) + 4)

    # 1) Bucket by file size (cheap)
    size_buckets: dict[int, list[str]] = defaultdict(list)
    for ap in abs_paths:
        p = Path(ap)
        try:
            size_buckets[p.stat().st_size].append(ap)
        except OSError:
            # If file is missing/unreadable, skip here (bad list should catch it)
            pass

    # Only hash buckets with collisions
    candidates: list[str] = []
    for _, bucket in size_buckets.items():
        if len(bucket) > 1:
            candidates.extend(bucket)

    # 2) Hash only collision candidates
    def _hash_abs(ap: str) -> tuple[str | None, str]:
        p = Path(ap)
        try:
            return sha256_file(p), ap
        except Exception:
            return None, ap

    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for h, ap in ex.map(_hash_abs, candidates):
            if h is not None:
                hash_to_paths[h].append(ap)

    rel_paths = [str(Path(x).relative_to(root_dir)) for x in abs_paths]

    dup_groups = {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}

    # 3) Build drop/keep lists (keep one representative per dup group)
    dup_set: set[str] = set()
    keep_set: set[str] = set(rel_paths)  # start with all, then drop duplicates

    for _, ps in dup_groups.items():
        ps_sorted = sorted(ps)
        keep_one = ps_sorted[0]
        keep_set.add(keep_one)
        for rp in ps_sorted[1:]:
            dup_set.add(rp)
            if rp in keep_set:
                keep_set.remove(rp)

    dup_files_list = sorted(dup_set)
    keep_files_list = sorted(keep_set)

    return dup_groups, dup_files_list, keep_files_list


def rgba_to_rgb_with_bg(img: Image.Image, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    background = Image.new("RGBA", img.size, bg + (255,))
    composed = Image.alpha_composite(background, img)
    return composed.convert("RGB")


def load_filtered_imagefolder(
    root_dir: str | Path,
    bad_files_path: str | Path | None = None,
    duplicate_groups_path: str | Path | None = None,
    ignore_files_path: str | Path | None = None,
    transform=None,
    loader=None,
):
    """
    Build an ImageFolder dataset and exclude files listed in special-case JSONs.

    Expected JSON formats:
      - bad_files.json: {"bad_files_list": [...], "total_file_num": 25000}
      - ignore_files_list.json: {"ignore_files_list": [...], "total_file_number": 24998}
      - duplicate_groups_exact.json: {"hash": [path1, path2, ...], ...}

    For duplicate groups, all but the first element in each group are excluded.
    """

    from torchvision.datasets import ImageFolder

    root_dir = Path(root_dir).resolve()
    ds = ImageFolder(root_dir, transform=transform, loader=loader)
    ds_total = len(ds.samples)

    if bad_files_path is None:
        bad_files_path = root_dir / "bad_files.json"
    if duplicate_groups_path is None:
        duplicate_groups_path = root_dir / "duplicate_groups_exact.json"
    if ignore_files_path is None:
        ignore_files_path = root_dir / "ignore_files_list.json"

    def _normalize_rel(path: str | Path) -> str | None:
        p = Path(path)
        if not p.is_absolute():
            p = root_dir / p
        try:
            return p.resolve().relative_to(root_dir).as_posix()
        except Exception:
            return None

    def _check_total(meta: dict, label: str, path: Path) -> None:
        total = meta.get("total_file_number", meta.get("total_file_num"))
        if total is None:
            warnings.warn(
                f"{label} missing total file count in {path}. Please update this list.",
                stacklevel=2,
            )
            return
        if total != ds_total:
            warnings.warn(
                f"{label} total ({total}) != dataset size ({ds_total}) for {path}. "
                "Please update this list.",
                stacklevel=2,
            )

    exclusion_set: set[str] = set()

    bad_path = Path(bad_files_path)
    if bad_path.exists():
        bad_meta = json.loads(bad_path.read_text(encoding="utf-8"))
        _check_total(bad_meta, "bad_files", bad_path)
        for item in bad_meta.get("bad_files_list", []):
            rel = _normalize_rel(item)
            if rel is not None:
                exclusion_set.add(rel)

    dup_path = Path(duplicate_groups_path)
    if dup_path.exists():
        dup_meta = json.loads(dup_path.read_text(encoding="utf-8"))
        _check_total(dup_meta, "duplicate_groups", dup_path)
        for _, group in dup_meta['dup_groups'].items():
            if not isinstance(group, list):
                continue
            for item in group[1:]:
                rel = _normalize_rel(item)
                if rel is not None:
                    exclusion_set.add(rel)

    ignore_path = Path(ignore_files_path)
    if ignore_path.exists():
        ignore_meta = json.loads(ignore_path.read_text(encoding="utf-8"))
        _check_total(ignore_meta, "ignore_files", ignore_path)
        for item in ignore_meta.get("ignore_files_list", []):
            rel = _normalize_rel(item)
            if rel is not None:
                exclusion_set.add(rel)

    filtered_samples = []
    for path, target in ds.samples:
        rel = _normalize_rel(path)
        if rel is None or rel not in exclusion_set:
            filtered_samples.append((path, target))

    ds.samples = filtered_samples
    ds.imgs = filtered_samples
    ds.targets = [target for _, target in filtered_samples]

    return ds, exclusion_set
