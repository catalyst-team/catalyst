from typing import Dict
from collections import OrderedDict

import pandas as pd

from catalyst.data.nlp.key_phrase import KeyPhrasesDataset
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    def __init__(self, config: Dict):
        super().__init__(config)

    def get_transforms(self, stage: str = None, mode: str = None):
        return []

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train = pd.read_json(
            "./nlp_key_phrase/input/train.jsonl",
            lines=True,
            orient="records",
        )
        val = pd.read_json(
            "./nlp_key_phrase/input/test.jsonl",
            lines=True,
            orient="records",
        )

        trainset = KeyPhrasesDataset(
            train["content"],
            train["tagged_attributes"],
        )
        testset = KeyPhrasesDataset(val["content"], val["tagged_attributes"])

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
