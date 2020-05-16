from typing import Dict
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from torch.utils.data import DataLoader  # noqa: F401
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from catalyst.contrib.data.nlp import LMDataset
from catalyst.dl import ConfigExperiment, utils


class Experiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self, config: Dict):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__(config)
        self.config = config
        self.tokenizer = None

    def get_transforms(self, stage: str = None, mode: str = None):
        """
        @TODO: Docs. Contribution is welcome
        """
        return []

    # noinspection PyMethodOverriding
    def get_datasets(
        self,
        stage: str,
        path_to_data: str,
        train_filename: str,
        text_field: str,
        max_sequence_length: int,
        **kwargs
    ):
        """
        @TODO: Docs. Contribution is welcome
        """
        datasets = OrderedDict()

        path_to_data = Path(path_to_data)

        train_df = pd.read_csv(path_to_data / train_filename)

        train_dataset = LMDataset(
            texts=train_df[text_field],
            max_seq_length=max_sequence_length,
            tokenizer=self.tokenizer,
        )
        datasets["train"] = train_dataset
        return datasets

    def get_loaders(
        self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        data_params = dict(self.stages_config[stage]["data_params"])
        model_name = data_params["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        collate_fn = DataCollatorForLanguageModeling(
            self.tokenizer
        ).collate_batch
        loaders_params = {"train": {"collate_fn": collate_fn}}
        loaders = utils.get_loaders_from_params(
            get_datasets_fn=self.get_datasets,
            initial_seed=self.initial_seed,
            stage=stage,
            loaders_params=loaders_params,
            **data_params,
        )

        return loaders

    def get_model(self, stage: str):
        """@TODO"""
        model_params = self._config["model_params"]
        model = AutoModelWithLMHead.from_pretrained(**model_params)
        return model
