from typing import Dict
from collections import OrderedDict

import pandas as pd

from catalyst.data.nlp.classify import ClassificationDataset
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.config = config

    def get_transforms(self, stage: str = None, mode: str = None):
        return []

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train_df = pd.read_csv(self.config['dataset_params']['path_to_train'])
        valid_df = pd.read_csv(self.config['dataset_params']['path_to_validation'])

        train_set = ClassificationDataset(
            texts=train_df[self.config['dataset_params']['text_field']],
            labels=train_df[self.config['dataset_params']['label_field']]
        )

        validation_set = ClassificationDataset(
            texts=valid_df[self.config['dataset_params']['text_field']],
            labels=valid_df[self.config['dataset_params']['label_field']]
        )

        datasets["train"] = train_set
        datasets["valid"] = validation_set

        return datasets
