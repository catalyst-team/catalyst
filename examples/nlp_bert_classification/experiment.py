from pathlib import Path
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
        
        path_to_data = Path(self.config['dataset_params']['path_to_data'])
        train_df = pd.read_csv(path_to_data / self.config['dataset_params']['train_filename'])
        valid_df = pd.read_csv(path_to_data / self.config['dataset_params']['validation_filename'])
        test_df = pd.read_csv(path_to_data / self.config['dataset_params']['test_filename'])

        max_seq_length = self.config['dataset_params']['max_sequence_length']

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
        #datasets["test"] = test_dataset

        return datasets

