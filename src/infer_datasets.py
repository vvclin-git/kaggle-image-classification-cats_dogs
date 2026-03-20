"""Dataset factories for inference-only datasets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAMPLE_SIZE = 256
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _path_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, f"{int(stem):012d}")
    return (1, stem.lower())


class UnlabeledImageDataset(Dataset):
    """Simple unlabeled image dataset for batched CLI inference."""

    def __init__(self, root_dir: str | Path, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.paths = sorted(
            (p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS),
            key=_path_sort_key,
        )
        if not self.paths:
            raise FileNotFoundError(f"No supported image files found under: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def build_unlabeled_dataset(root_dir: str | Path) -> UnlabeledImageDataset:
    """Factory for an unlabeled image folder used for inference."""

    tf = transforms.Compose(
        [
            transforms.Resize(SAMPLE_SIZE),
            transforms.CenterCrop(SAMPLE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return UnlabeledImageDataset(root_dir, transform=tf)
