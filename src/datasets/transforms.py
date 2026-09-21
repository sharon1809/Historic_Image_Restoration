import random
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image, ImageDraw


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


class PairedSyntheticScratches:
    """
    Simulates historical damage by randomly drawing lines (scratches) 
    onto the damaged image, leaving the clean image untouched.
    This forces the model to learn to explicitly erase this damage.
    """
    def __init__(self, probability: float = 0.3, max_scratches: int = 10):
        self.probability = probability
        self.max_scratches = max_scratches

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        if random.random() > self.probability:
            return damaged, clean

        # We must operate on a copy to avoid mutating original references if any
        damaged_copy = damaged.copy()
        draw = ImageDraw.Draw(damaged_copy)
        
        width, height = damaged_copy.size
        num_scratches = random.randint(1, self.max_scratches)
        
        for _ in range(num_scratches):
            # Random coordinates for the line
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            
            # Scratches are usually short-ish relative to image size
            length = random.randint(10, min(width, height) // 3)
            angle = random.uniform(0, 3.14159) # roughly 0 to 180 degrees in radians
            
            import math
            x2 = x1 + int(length * math.cos(angle))
            y2 = y1 + int(length * math.sin(angle))
            
            # Scratches are usually white/light grey or black/dark brown
            is_white_scratch = random.random() > 0.5
            if is_white_scratch:
                color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            else:
                color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
                
            # Random thickness
            width_line = random.randint(1, 3)
            
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width_line)

        return damaged_copy, clean


class PairedTransform:
    """
    Apply paired spatial transformations followed by tensor conversion.
    """

    def __init__(self, crop_size: int = 256, training: bool = True, use_synthetic_scratches: bool = False):
        self.crop = PairedRandomCrop(crop_size) if training else PairedCenterCrop(crop_size)
        self.flip = PairedHorizontalFlip() if training else None
        self.scratch = PairedSyntheticScratches() if (training and use_synthetic_scratches) else None
        self.to_tensor = transforms.ToTensor()

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        damaged, clean = self.crop(damaged, clean)

        if self.flip is not None:
            damaged, clean = self.flip(damaged, clean)
            
        if self.scratch is not None:
            damaged, clean = self.scratch(damaged, clean)

        damaged = self.to_tensor(damaged)
        clean = self.to_tensor(clean)

        return damaged, clean