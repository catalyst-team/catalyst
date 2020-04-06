from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from catalyst.dl import ConfigExperiment

from .dataset import SegmentationDataset


class Experiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    def get_datasets(
        self,
        stage: str,
        image_path: str,
        mask_path: str,
        valid_size: float,
        **kwargs
    ):
        """
        @TODO: Docs. Contribution is welcome
        """
        _images = np.array(sorted(Path(image_path).glob("*.jpg")))
        _masks = np.array(sorted(Path(mask_path).glob("*.gif")))

        _indices = np.arange(len(_images))

        train_indices, valid_indices = train_test_split(
            _indices,
            test_size=valid_size,
            random_state=self.initial_seed,
            shuffle=True,
        )

        datasets = OrderedDict()
        for mode, indices in zip(
            ["train", "valid"], [train_indices, valid_indices]
        ):
            datasets[mode] = SegmentationDataset(
                images=_images[indices].tolist(),
                masks=_masks[indices].tolist(),
                transforms=self.get_transforms(stage=stage, dataset=mode),
            )

        return datasets
