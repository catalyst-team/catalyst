from pathlib import Path
from typing import List

from skimage.io import imread as gif_imread

from torch.utils.data import Dataset

from catalyst import utils


class SegmentationDataset(Dataset):
    """Dataset for segmentation tasks
    Returns a dict with ``image``, ``mask`` and ``filename`` keys
    """
    def __init__(
        self,
        images: List[Path],
        masks: List[Path] = None,
        transforms=None
    ) -> None:
        """
        Args:
            images (List[Path]): list of paths to the images
            masks (List[Path]): list of paths to the masks
                (names must be the same as in images)
            transforms: optional dict transforms
        """
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Main method"""
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(result)

        result["filename"] = image_path.name

        return result
