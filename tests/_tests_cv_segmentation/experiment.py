# flake8: noqa
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from catalyst.dl import ConfigExperiment
from tests._tests_cv_segmentation.dataset import SegmentationDataset  # noqa: WPS436


class Experiment(ConfigExperiment):
    """Docs? Contribution is welcome."""

    def get_datasets(
        self, stage: str, image_path: str, mask_path: str, valid_size: float, **kwargs
    ):
        """Docs? Contribution is welcome."""
        images = np.array(sorted(Path(image_path).glob("*.jpg")))
        masks = np.array(sorted(Path(mask_path).glob("*.gif")))
        indices = np.arange(len(images))

        train_indices, valid_indices = train_test_split(
            indices, test_size=valid_size, random_state=self.seed, shuffle=True,
        )

        datasets = OrderedDict()
        for mode, split_indices in zip(["train", "valid"], [train_indices, valid_indices]):
            datasets[mode] = SegmentationDataset(
                images=images[split_indices].tolist(),
                masks=masks[split_indices].tolist(),
                transforms=self.get_transforms(stage=stage, dataset=mode),
            )

        return datasets
