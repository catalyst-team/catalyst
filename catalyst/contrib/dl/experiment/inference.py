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
    ) -> OrderedDict[str, Dataset]:
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
        input_key = data_params["input_key"]

        dataset = OrderedDict(
            infer=TiledImageDataset(
                image_path, tile_size, tile_step, input_key=input_key
            )
        )

        return dataset

    def get_callbacks(self, stage: str) -> OrderedDict[Callback]:
        """
        Updates callbacks with additional one for tiled inference.
        """
        callbacks = super().get_callbacks(stage)
        data_params = dict(self.stages_config[stage]["data_params"])

        save_path = data_params["save_path"]
        image_path = data_params["image_path"]
        n_channels = data_params["n_channels"]
        tile_size = data_params["tile_size"]
        tile_step = data_params["tile_step"]
        output_key = data_params["output_key"]

        image = skimage.io.imread(image_path)
        *image_size, _ = image.shape

        tiled_inference_callback = TiledInferenceCallback(
            save_path,
            image_size,
            n_channels,
            tile_size,
            tile_step,
            output_key=output_key,
        )
        callbacks.update(tiled_inference=tiled_inference_callback)

        return callbacks
