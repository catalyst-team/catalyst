#!/usr/bin/env python
# flake8: noqa
from typing import Any, Optional
from argparse import ArgumentParser, RawTextHelpFormatter

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from catalyst import dl

from src import E2E, parse_params


class CustomRunner(dl.IRunner):
    def __init__(self, logdir: str, engine: str, **engine_params: Any):
        super().__init__()
        self._logdir = logdir
        self._engine = engine
        self._engine_params = engine_params

    def get_engine(self):
        return E2E[self._engine](**self._engine_params)

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def num_epochs(self) -> int:
        return 10

    def get_loaders(
        self,
    ):
        datasets = load_dataset("glue", "sst2")
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        encoded_datasets = datasets.map(
            lambda examples: tokenizer(
                examples["sentence"],
                max_length=128,
                truncation=True,
                padding="max_length",
            ),
            batched=True,
        )
        encoded_datasets = encoded_datasets.map(lambda x: {"labels": x["label"]})
        encoded_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        train_data = encoded_datasets["train"]
        valid_data = encoded_datasets["validation"]

        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.engine.num_processes,
                rank=self.engine.process_index,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                valid_data,
                num_replicas=self.engine.num_processes,
                rank=self.engine.process_index,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        self.train_loader_len = len(
            DataLoader(train_data, batch_size=64, sampler=train_sampler)
        )

        return {
            "train": DataLoader(train_data, batch_size=64, sampler=train_sampler),
            "valid": DataLoader(valid_data, batch_size=32, sampler=valid_sampler),
        }

    def get_model(
        self,
    ):
        model = (
            self.model
            if self.model is not None
            else AutoModelForSequenceClassification.from_pretrained("albert-base-v2")
        )
        return model

    def get_criterion(
        self,
    ):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=3e-5)

    def get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.05 * self.train_loader_len) * self.num_epochs,
            num_training_steps=self.train_loader_len * self.num_epochs,
        )
        return scheduler

    def get_callbacks(
        self,
    ):
        return {
            "criterion": dl.CriterionCallback(
                input_key="logits", target_key="labels", metric_key="loss"
            ),
            "backward": dl.BackwardCallback(metric_key="loss"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "scheduler": dl.SchedulerCallback(
                loader_key="valid", metric_key="loss", mode="batch"
            ),
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="labels", topk=(1,)
            ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="accuracy01",
                minimize=False,
                topk=1,
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def handle_batch(self, batch):
        outputs = self.model(**batch)

        self.batch = {
            "features": batch["input_ids"],
            "labels": batch["labels"],
            "logits": outputs.logits,
        }


if __name__ == "__main__":
    kwargs, _ = parse_params("albert")
    runner = CustomRunner(**kwargs)
    runner.run()
