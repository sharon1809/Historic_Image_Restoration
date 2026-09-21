from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from .transforms import PairedTransform


class OpenPhotoRestoreDataset(Dataset):
    """
    PyTorch Dataset wrapper for the OpenPhoto Restore Dataset.

    Returns:
        {
            "damaged": Tensor [3, H, W],
            "clean": Tensor [3, H, W]
        }
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        crop_size: int = 256,
        training: bool = True,
        use_synthetic_scratches: bool = False,
    ):
        self.dataset = hf_dataset

        self.transform = PairedTransform(
            crop_size=crop_size,
            training=training,
            use_synthetic_scratches=use_synthetic_scratches
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        damaged = sample["damaged_image"].convert("RGB")
        clean = sample["pristine_image"].convert("RGB")

        damaged, clean = self.transform(damaged, clean)

        return {
            "damaged": damaged,
            "clean": clean,
        }