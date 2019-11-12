import pandas as pd
from typing import Dict
from collections import OrderedDict
from catalyst.data.nlp.classify import TextClassificationDataset
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
        test_df = pd.read_csv(self.config['dataset_params']['path_to_test'])

        max_seq_length = self.config['model_params']['max_sequence_length']

        train_dataset = TextClassificationDataset(
            texts=train_df['text'],
            labels=train_df['label'],
            label_dict=None,
            max_seq_length=max_seq_length)

        valid_dataset = TextClassificationDataset(
            texts=valid_df['text'],
            labels=valid_df['label'],
            label_dict=train_dataset.label_dict,
            max_seq_length=max_seq_length)

        test_dataset = TextClassificationDataset(
            texts=test_df['text'],
            labels=None,
            label_dict=None,
            max_seq_length=max_seq_length)

        datasets["train"] = train_dataset
        datasets["valid"] = valid_dataset
        datasets["test"] = test_dataset

        return datasets

