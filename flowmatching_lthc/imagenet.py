"""ImageNet-256 dataset helpers.

Supports standard ImageFolder directories and EDM/REPA-style zip datasets with
`dataset.json` labels. Images are returned as uint8 tensors so training can use
in-place normalization on GPU.
"""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms


def center_crop_arr(pil_image, image_size):
    # Matches the ADM/EDM center-crop behavior used by Modified_DiT.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ZipImageNet256(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = str(zip_path)
        self.transform = transform
        self.zip = None
        with zipfile.ZipFile(self.zip_path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")))
            if "dataset.json" in zf.namelist():
                labels = dict(json.loads(zf.read("dataset.json"))["labels"])
            else:
                labels = self._labels_from_paths(names)
        self.names = names
        self.labels = [int(labels[n.replace('\\', '/')]) for n in names]

    @staticmethod
    def _labels_from_paths(names):
        roots = sorted({n.split('/')[0] for n in names if '/' in n})
        if len(roots) != 1000:
            raise RuntimeError("zip has no dataset.json and does not look like class-folder ImageNet")
        class_to_idx = {c: i for i, c in enumerate(roots)}
        return {n: class_to_idx[n.split('/')[0]] for n in names}

    def _zipfile(self):
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.zip_path)
        return self.zip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with self._zipfile().open(self.names[idx], 'r') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transform(image_size=256, train=True):
    ops = [transforms.Lambda(lambda img: center_crop_arr(img, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.PILToTensor())
    return transforms.Compose(ops)


def build_dataset(data_path, split='train', image_size=256):
    data_path = Path(data_path)
    transform = build_transform(image_size=image_size, train=(split == 'train'))
    split_dir = data_path / split
    if split_dir.is_dir():
        return datasets.ImageFolder(split_dir, transform=transform)
    zip_candidates = [data_path / f'{split}.zip', data_path / 'images.zip']
    for z in zip_candidates:
        if z.is_file():
            return ZipImageNet256(z, transform=transform)
    raise FileNotFoundError(f"no ImageFolder split or zip found under {data_path}")


def build_loader(dataset, batch_size, num_workers=12, distributed=False, rank=0, world_size=1, seed=0, drop_last=True):
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
