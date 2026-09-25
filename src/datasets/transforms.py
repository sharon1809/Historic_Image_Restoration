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
    Simulates historical damage by randomly drawing lines, curves, and jagged scratches
    onto the damaged image, leaving the clean image untouched.
    """
    def __init__(self, probability: float = 1.0, max_scratches: int = 15):
        self.probability = probability
        self.max_scratches = max_scratches

    def _draw_organic_scratch(self, draw, width, height):
        import math
        
        # Scratch properties
        is_white = random.random() > 0.5
        base_color = random.randint(200, 255) if is_white else random.randint(0, 50)
        
        # Mix of subtle and severe
        initial_opacity = random.randint(100, 255)
        
        # Predominantly thin, occasionally thick
        thickness = random.choices([2, 5, 10, 15, 20], weights=[0.2, 0.3, 0.2, 0.2, 0.1])[0]
        
        # Start point
        x, y = float(random.randint(0, width)), float(random.randint(0, height))
        
        # Vary length substantially (from tiny flecks to long meandering scratches)
        num_segments = random.randint(10, 100)
        
        # Base direction
        angle = random.uniform(0, 2 * math.pi)
        
        # Angular momentum to create smooth, meandering curves rather than jagged polygons
        angle_momentum = random.uniform(-0.1, 0.1)
        
        for i in range(num_segments):
            # Calculate current opacity (fading out towards the end)
            fade_factor = 1.0 - (i / num_segments)
            current_opacity = int(initial_opacity * fade_factor)
            
            # If it's too transparent, stop drawing early
            if current_opacity < 10:
                break
                
            color = (base_color, base_color, base_color, current_opacity)
            
            # Very small segments (1-3 pixels) for smooth organic curvature
            segment_length = random.uniform(1.0, 3.0)
            
            # Update angle with momentum and a little bit of noise
            angle += angle_momentum + random.uniform(-0.05, 0.05)
            
            # Occasionally the scratch might "break" or change momentum abruptly (jagged)
            if random.random() < 0.05:
                angle_momentum = random.uniform(-0.2, 0.2)
                
            nx = x + (segment_length * math.cos(angle))
            ny = y + (segment_length * math.sin(angle))
            
            draw.line([(int(x), int(y)), (int(nx), int(ny))], fill=color, width=thickness)
            
            x, y = nx, ny

    def __call__(self, damaged: Image.Image, clean: Image.Image):
        if random.random() > self.probability:
            return damaged, clean

        # Create an RGBA copy of damaged for alpha compositing
        damaged_rgba = damaged.convert("RGBA")
        overlay = Image.new("RGBA", damaged_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = damaged.size
        # Wide variation in number of scratches
        num_scratches = random.randint(2, self.max_scratches)
        
        for _ in range(num_scratches):
            self._draw_organic_scratch(draw, width, height)

        # Composite the transparent scratches over the original image
        damaged_out = Image.alpha_composite(damaged_rgba, overlay).convert("RGB")
        return damaged_out, clean


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