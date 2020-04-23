from collections import OrderedDict

import skimage.io

from torch.utils.data import Dataset

from catalyst.contrib.data.cv.datasets import TiledImageDataset
from catalyst.contrib.dl.callbacks.inference import TiledInferenceCallback
from catalyst.core import Callback
from catalyst.dl.experiment import ConfigExperiment


class TiledInferenceExperiment(ConfigExperiment):
    """
    Experiment class to use for tiled inference.
    """

    def get_datasets(
        self, stage: str, epoch: int = None, **data_params
    ) -> "OrderedDict[str, Dataset]":
        """
        Defines dataset with tiles of huge image.
        """
        if not stage.startswith("infer"):
            raise ValueError(
                f"Only inference stage is supported in "
                f"{self.__class__.__name__}."
            )

        image_path = data_params["image_path"]
        tile_size = data_params["tile_size"]
        tile_step = data_params["tile_step"]
        input_key = data_params.get("input_key", "features")

        dataset = OrderedDict(
            infer=TiledImageDataset(
                image_path,
                tile_size,
                tile_step,
                input_key=input_key,
                transform=self.get_transforms(stage=stage, dataset="infer"),
            )
        )

        return dataset

    def get_callbacks(self, stage: str) -> "OrderedDict[Callback]":
        """
        Updates callbacks with additional one for tiled inference.
        """
        callbacks = super().get_callbacks(stage)
        data_params = dict(self.stages_config[stage]["data_params"])

        save_dir = data_params["save_dir"]
        image_path = data_params["image_path"]
        num_classes = data_params["num_classes"]
        tile_size = data_params["tile_size"]
        tile_step = data_params["tile_step"]
        threshold = data_params.get("threshold", 0.5)
        output_key = data_params.get("output_key", "logits")
        mask_key = data_params.get("mask_key", "mask")

        image = skimage.io.imread(image_path)
        *image_size, _ = image.shape

        tiled_inference_callback = TiledInferenceCallback(
            save_dir,
            image_size,
            num_classes,
            tile_size,
            tile_step,
            threshold=threshold,
            output_key=output_key,
            mask_key=mask_key,
        )
        callbacks.update(tiled_inference=tiled_inference_callback)

        return callbacks
