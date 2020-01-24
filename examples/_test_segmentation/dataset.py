from pathlib import Path
from typing import List

from skimage.io import imread as gif_imread

from torch.utils.data import Dataset

from catalyst import utils


class SegmentationDataset(Dataset):
    def __init__(
        self, images: List[Path], masks: List[Path] = None, transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result
