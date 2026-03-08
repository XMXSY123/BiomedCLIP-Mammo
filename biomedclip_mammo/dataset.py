# -*- coding: utf-8 -*-
"""乳腺图像数据集：图-文对 (image, text) 用于 BiomedCLIP 对比学习。"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

from .metadata_utils import build_image_to_text_map, image_id_and_view_from_path


def get_train_augment():
    """训练时图像增强：随机小角度旋转、随机水平翻转（验证/测试不做增强）。"""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.RandomRotation(15, fill=0),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    except Exception:
        return None


class MammoImageTextDataset(Dataset):
    """图-文对比学习：每样本 (image_tensor, text_string)，text 由 Metadata D 列病灶组成句子。"""

    def __init__(self, image_dir: str, metadata_path: str, preprocess=None, augment=None, samples=None):
        self.image_dir = Path(image_dir)
        self.preprocess = preprocess
        self.augment = augment
        if samples is not None:
            self.samples = list(samples)
        else:
            self.text_map = build_image_to_text_map(metadata_path)
            self.samples = []
            for path in self.image_dir.rglob("*.png"):
                try:
                    rel = path.relative_to(self.image_dir)
                except ValueError:
                    continue
                pid, view = image_id_and_view_from_path(str(rel))
                key = f"{pid}|{view}" if view else pid
                if key not in self.text_map:
                    if pid not in self.text_map:
                        continue
                    key = pid
                text = self.text_map[key]
                self.samples.append((str(path), text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.augment is not None:
            img = self.augment(img)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, text


def get_dataloaders_image_text(
    image_dir: str,
    metadata_path: str,
    preprocess,
    batch_size: int = 16,
    num_workers: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    train_augment: bool = True,
):
    """图-文对比学习用 DataLoader：每 batch (images, list_of_text_strings)。train_augment=True 时对训练集做随机旋转、随机水平翻转。"""
    from torch.utils.data import DataLoader, random_split
    full = MammoImageTextDataset(image_dir, metadata_path, preprocess=preprocess, augment=None)
    n = len(full)
    n_test = max(0, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio)) if n_test == 0 else int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = n - n_test - 1
    gen = torch.Generator().manual_seed(seed)
    train_kw = {"num_workers": num_workers, "pin_memory": True}
    loader_kw = {"num_workers": num_workers}
    if num_workers > 0:
        train_kw["persistent_workers"] = True
        loader_kw["persistent_workers"] = True
    augment = get_train_augment() if train_augment else None
    if n_test > 0:
        train_sub, val_sub, test_sub = random_split(full, [n_train, n_val, n_test], generator=gen)
        train_indices = train_sub.indices
        val_indices = val_sub.indices
        test_indices = test_sub.indices
    else:
        train_sub, val_sub = random_split(full, [n_train, n_val], generator=gen)
        train_indices = train_sub.indices
        val_indices = val_sub.indices
        test_indices = None
    train_samples = [full.samples[i] for i in train_indices]
    val_samples = [full.samples[i] for i in val_indices]
    train_ds = MammoImageTextDataset(image_dir, metadata_path, preprocess=preprocess, augment=augment, samples=train_samples)
    val_ds = MammoImageTextDataset(image_dir, metadata_path, preprocess=preprocess, augment=None, samples=val_samples)
    if n_test > 0:
        test_samples = [full.samples[i] for i in test_indices]
        test_ds = MammoImageTextDataset(image_dir, metadata_path, preprocess=preprocess, augment=None, samples=test_samples)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **train_kw)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **train_kw)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
        return train_loader, val_loader, None
