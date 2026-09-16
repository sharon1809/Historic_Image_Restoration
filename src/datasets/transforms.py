import random
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image


class PairedRandomCrop:
    """
    Apply the same random crop to a damaged image and its clean target.
    """

    def __init__(self, size: int | Tuple[int, int]):
        if isinstance(size, int):
            self.crop_height = size
            self.crop_width = size
        else:
            self.crop_height, self.crop_width = size

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        width, height = damaged.size

        if clean.size != damaged.size:
            raise ValueError(
                f"Image sizes do not match: "
                f"damaged={damaged.size}, clean={clean.size}"
            )

        if height < self.crop_height or width < self.crop_width:
            raise ValueError(
                f"Image size {damaged.size} is smaller than "
                f"crop size {(self.crop_width, self.crop_height)}"
            )

        top = random.randint(0, height - self.crop_height)
        left = random.randint(0, width - self.crop_width)

        damaged = damaged.crop(
            (left, top, left + self.crop_width, top + self.crop_height)
        )

        clean = clean.crop(
            (left, top, left + self.crop_width, top + self.crop_height)
        )

        return damaged, clean


class PairedHorizontalFlip:
    """
    Apply the same random horizontal flip to both images.
    """

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        if random.random() < self.probability:
            damaged = damaged.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return damaged, clean


class PairedCenterCrop:
    """
    Apply the same deterministic center crop to both images.
    """

    def __init__(self, size: int | Tuple[int, int]):
        if isinstance(size, int):
            self.crop_height = size
            self.crop_width = size
        else:
            self.crop_height, self.crop_width = size

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        width, height = damaged.size

        if clean.size != damaged.size:
            raise ValueError(
                f"Image sizes do not match: "
                f"damaged={damaged.size}, clean={clean.size}"
            )

        if height < self.crop_height or width < self.crop_width:
            raise ValueError(
                f"Image size {damaged.size} is smaller than "
                f"crop size {(self.crop_width, self.crop_height)}"
            )

        top = (height - self.crop_height) // 2
        left = (width - self.crop_width) // 2

        damaged = damaged.crop(
            (left, top, left + self.crop_width, top + self.crop_height)
        )

        clean = clean.crop(
            (left, top, left + self.crop_width, top + self.crop_height)
        )

        return damaged, clean


class PairedTransform:
    """
    Apply paired spatial transformations followed by tensor conversion.
    """

    def __init__(self, crop_size: int = 256, training: bool = True):
        self.crop = PairedRandomCrop(crop_size) if training else PairedCenterCrop(crop_size)
        self.flip = PairedHorizontalFlip() if training else None
        self.to_tensor = transforms.ToTensor()

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        damaged, clean = self.crop(damaged, clean)

        if self.flip is not None:
            damaged, clean = self.flip(damaged, clean)

        damaged = self.to_tensor(damaged)
        clean = self.to_tensor(clean)

        return damaged, clean