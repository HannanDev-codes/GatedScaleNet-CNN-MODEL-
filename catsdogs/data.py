"""Loading images and turning them into numbers the CNN can read.

Expected folder layout (this is exactly how your zip is organised):

    data/
      train/cat/*.jpg
      train/dog/*.jpg
      val/cat/*.jpg
      val/dog/*.jpg

The folder name is the label. Cat comes first alphabetically, so cat=0 and dog=1.
"""
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard statistics used to centre colour values (needed for pretrained models, harmless otherwise)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(img_size: int = 128):
    """Returns (train_transform, eval_transform).

    Training images are randomly flipped / rotated / re-coloured every time they are shown.
    That is "data augmentation": with only ~275 images it gives the model many more
    different-looking examples so it learns the real pattern instead of memorising pictures.
    Validation / prediction images are only resized, so results are fair and repeatable.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def get_datasets(data_dir: str, img_size: int = 128):
    root = Path(data_dir)
    for split in ("train", "val"):
        if not (root / split).is_dir():
            raise FileNotFoundError(
                f"Could not find '{root / split}'. Put the dataset in '{data_dir}' "
                "so it contains train/cat, train/dog, val/cat and val/dog."
            )
    train_tf, eval_tf = get_transforms(img_size)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise ValueError(f"train classes {train_ds.classes} != val classes {val_ds.classes}")
    return train_ds, val_ds


def get_loaders(data_dir: str, img_size: int = 128, batch_size: int = 32, workers: int = 2):
    train_ds, val_ds = get_datasets(data_dir, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_ds, val_ds, train_loader, val_loader


def class_weights(targets, num_classes: int) -> torch.Tensor:
    """Your data has ~2x more dogs than cats. Without correction the model could cheat by
    guessing 'dog' a lot. These weights make a mistake on a cat count about 2x more."""
    counts = Counter(targets)
    total = sum(counts.values())
    return torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)], dtype=torch.float
    )
