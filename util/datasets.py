# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import csv
import os
import PIL

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _has_class_subdirs(root):
    """True if `root` contains at least one subdirectory (ImageFolder layout)."""
    return any(entry.is_dir() for entry in os.scandir(root))


def _find_classes(train_dir):
    """Replicate torchvision ImageFolder's class ordering (sorted dir names).

    Returns a {wnid: index} mapping identical to the one ImageFolder builds for
    the train split, so validation labels line up with training labels.
    """
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders found in {train_dir}")
    return {cls: idx for idx, cls in enumerate(classes)}


def _locate_val_solution_csv(data_path):
    """Find LOC_val_solution.csv (Kaggle CLS-LOC) relative to the data path.

    data_path is typically .../ILSVRC/Data/CLS-LOC; the csv sits at the kaggle
    root, so we walk a few parent levels up looking for it.
    """
    candidates = [
        os.path.join(data_path, "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "..", "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "..", "..", "LOC_val_solution.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


class FlatImageNetVal(torch.utils.data.Dataset):
    """ImageNet validation set stored as flat ``ILSVRC2012_val_*.JPEG`` files.

    This is the layout shipped by the Kaggle "imagenet-object-localization-challenge"
    (a.k.a. CLS-LOC) dataset: ``train/`` has per-class WNID subfolders but ``val/``
    is a single flat directory of images. Labels are read from
    ``LOC_val_solution.csv`` and mapped through ``class_to_idx`` (derived from the
    train folders) so indices match torchvision ImageFolder on the train split.
    """

    def __init__(self, val_dir, class_to_idx, solution_csv, transform=None, loader=default_loader):
        self.val_dir = val_dir
        self.transform = transform
        self.loader = loader

        # csv rows: "ImageId,PredictionString" where PredictionString starts with the WNID.
        img_to_wnid = {}
        with open(solution_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                image_id, pred = row[0], row[1]
                img_to_wnid[image_id] = pred.split()[0]

        self.samples = []
        missing = 0
        for image_id, wnid in img_to_wnid.items():
            if wnid not in class_to_idx:
                continue
            path = os.path.join(val_dir, image_id + ".JPEG")
            if not os.path.isfile(path):
                missing += 1
                continue
            self.samples.append((path, class_to_idx[wnid]))

        if not self.samples:
            raise RuntimeError(
                f"No validation images matched between {val_dir} and {solution_csv}"
            )
        self.samples.sort()
        if missing:
            print(f"[FlatImageNetVal] warning: {missing} csv entries had no image on disk")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_imagenet_val(args, transform):
    """Build the ImageNet val dataset, supporting both the class-subfolder and the
    flat (Kaggle CLS-LOC) layouts. Used by both finetuning and linear probing."""
    val_root = os.path.join(args.data_path, "val")
    if _has_class_subdirs(val_root):
        return datasets.ImageFolder(val_root, transform=transform)

    class_to_idx = _find_classes(os.path.join(args.data_path, "train"))
    csv_path = _locate_val_solution_csv(args.data_path)
    if csv_path is None:
        raise FileNotFoundError(
            f"The validation folder {val_root} is flat (ILSVRC2012_val_*.JPEG) but no "
            "LOC_val_solution.csv was found relative to --data_path "
            f"({args.data_path})."
        )
    dataset = FlatImageNetVal(val_root, class_to_idx, csv_path, transform=transform)
    print(f"FlatImageNetVal: {len(dataset)} images, {len(class_to_idx)} classes "
          f"(labels from {csv_path})")
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if is_train:
        root = os.path.join(args.data_path, 'train')
        dataset = datasets.ImageFolder(root, transform=transform)
        print(dataset)
        return dataset

    dataset = build_imagenet_val(args, transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
