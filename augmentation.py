"""
see https://pytorch.org/vision/0.15/transforms.html

"""
import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class SegmentationCompose():
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        for t in self.transforms:
            image, label = t(image, label)

        return image, label


class SegmentationRandomCrop():
    def __init__(self, crop_size) -> None:
        self.size = crop_size

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        assert self.size < image.shape[-1] and self.size < image.shape[-2]
        left = random.choice(range(image.shape[-1]-self.size))
        top = random.choice(range(image.shape[-2]-self.size))

        im_cropped = TF.crop(image, top, left, self.size, self.size)
        lab_cropped = TF.crop(label, top, left, self.size, self.size)

        return im_cropped, lab_cropped


class SegmentationCenterCrop():
    def __init__(self, crop_size) -> None:
        self.size = crop_size

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        im_cropped = TF.center_crop(image, self.size)
        lab_cropped = TF.center_crop(label, self.size)

        return im_cropped, lab_cropped


def to_tensor(image: torch.Tensor, label: torch.Tensor):
    return TF.to_tensor(image), TF.to_tensor(label)


def default(train_or_test, *, crop_size: int, **kwargs):
    if train_or_test == 'train':
        return SegmentationCompose([
            to_tensor,
            SegmentationRandomCrop(crop_size)
        ])
    elif train_or_test == 'test':
        return SegmentationCompose([
            to_tensor,
            SegmentationCenterCrop(crop_size)
        ])
    else:
        raise ValueError(train_or_test)
