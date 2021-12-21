#!/usr/bin/env python
# flake8: noqa
from typing import Optional
from argparse import ArgumentParser, RawTextHelpFormatter

from common import E2E, parse_ddp_params

from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from catalyst import dl


class CustomRunner(dl.IRunner):
    def __init__(self, logdir: str, engine: str, engine_params: Optional[dict] = None):
        super().__init__()
        self._logdir = logdir
        self._engine = engine
        self._engine_params = engine_params or {}

    def get_engine(self):
        return E2E[self._engine](**self._engine_params)

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 10

    def get_loaders(self, stage: str):
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
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                valid_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        self.train_loader_len = len(DataLoader(train_data, batch_size=64, sampler=train_sampler))

        return {
            "train": DataLoader(train_data, batch_size=64, sampler=train_sampler),
            "valid": DataLoader(valid_data, batch_size=32, sampler=valid_sampler),
        }

    def get_model(self, stage: str):
        model = (
            self.model
            if self.model is not None
            else AutoModelForSequenceClassification.from_pretrained("albert-base-v2")
        )
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        return optim.Adam(model.parameters(), lr=3e-5)

    def get_scheduler(self, stage: str, optimizer):
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.05 * self.train_loader_len) * self.stage_epoch_len,
            num_training_steps=self.train_loader_len * self.stage_epoch_len,
        )
        return scheduler

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                input_key="logits", target_key="labels", metric_key="loss"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss", mode="batch"),
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="labels", topk_args=(1,)
            ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="accuracy01",
                minimize=False,
                save_n_best=1,
            ),
            # "tqdm": dl.TqdmCallback(),
        }

    def handle_batch(self, batch):
        outputs = self.model(**batch)

        self.batch = {
            "features": batch["input_ids"],
            "labels": batch["labels"],
            "logits": outputs.logits,
        }


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--engine", type=str, choices=list(E2E.keys()))
    args, unknown_args = parser.parse_known_args()
    args.logdir = args.logdir or f"logs_albert_{args.engine}".replace("-", "_")
    if args.engine in {"ddp", "amp-ddp", "apex-ddp", "ds-ddp", "fs-ddp", "fs-ddp-amp", "fs-fddp"}:
        engine_params, _ = parse_ddp_params(unknown_args)

        # fix for DeepSpeed engine since is does not support batchnorm synchonization
        if args.engine == "ds-ddp":
            engine_params.pop("sync_bn")
    else:
        engine_params = None

    runner = CustomRunner(args.logdir, args.engine, engine_params)
    runner.run()
