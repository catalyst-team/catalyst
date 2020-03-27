from collections import OrderedDict

from torch.utils.data import Dataset

from catalyst.contrib.data.cv.datasets import TiledImageDataset
from catalyst.dl.experiment import ConfigExperiment


class TiledInferenceExperiment(ConfigExperiment):
    def get_datasets(
        self, stage: str, epoch: int = None, **data_params
    ) -> OrderedDict[str, Dataset]:

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
