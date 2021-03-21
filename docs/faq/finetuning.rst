Finetuning (multistage runs)
==============================================================================

Suppose you have a large pretrained network you want to adapt for your task.
Most common approach in this case would be to finetune the network on our dataset –
use the large network as an encoder for your small classification head and train only this head.

Nevertheless to get the best possible results, it's better to use two-stage approach:
    - freeze the encoder network and train only the classification head during the first stage
    - unfreeze the whole network and tune encoder with head on the second stage

Thanks to Catalyst Runner API,
it's quite easy to create such complex pipeline with a few line of code:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl, utils
    from catalyst.contrib.datasets import MNIST
    from catalyst.data.transforms import ToTensor


    class CustomRunner(dl.IRunner):
        def __init__(self, logdir, device):
            # you could add all required extra params during Runner initialization
            # for our case, let's customize ``logdir`` and ``engine`` for the runs
            super().__init__()
            self._logdir = logdir
            self._device = device

        def get_engine(self):
            return dl.DeviceEngine(self._device)

        def get_loggers(self):
            return {
                "console": dl.ConsoleLogger(),
                "csv": dl.CSVLogger(logdir=self._logdir),
                "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
            }

        @property
        def stages(self):
            # suppose we have 2 stages:
            # 1st - with freezed encoder
            # 2nd with unfreezed whole network
            return ["train_freezed", "train_unfreezed"]

        def get_stage_len(self, stage: str) -> int:
            return 3

        def get_loaders(self, stage: str):
            loaders = {
                "train": DataLoader(
                    MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                    batch_size=32
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                    batch_size=32
                ),
            }
            return loaders

        def get_model(self, stage: str):
            # the logic here is quite straightforward:
            # we create the model on the fist stage
            # and reuse it during next stages
            model = (
                self.model
                if self.model is not None
                else nn.Sequential(
                    nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
                )
            )
            if stage == "train_freezed":
                # 1st stage
                # freeze layer
                utils.set_requires_grad(model[1], False)
            else:
                # 2nd stage
                utils.set_requires_grad(model, True)
            return model

        def get_criterion(self, stage: str):
            return nn.CrossEntropyLoss()

        def get_optimizer(self, stage: str, model):
            # we could also define different components for the different stages
            if stage == "train_freezed":
                return optim.Adam(model.parameters(), lr=1e-3)
            else:
                return optim.SGD(model.parameters(), lr=1e-1)

        def get_scheduler(self, stage: str, optimizer):
            return None

        def get_callbacks(self, stage: str):
            return {
                "criterion": dl.CriterionCallback(
                    metric_key="loss", input_key="logits", target_key="targets"
                ),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
                # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
                "accuracy": dl.AccuracyCallback(
                    input_key="logits", target_key="targets", topk_args=(1, 3, 5)
                ),
                "classification": dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=10
                ),
                # catalyst[ml] required
                # "confusion_matrix": dl.ConfusionMatrixCallback(
                #     input_key="logits", target_key="targets", num_classes=10
                # ),
                "checkpoint": dl.CheckpointCallback(
                    self._logdir,
                    loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
                ),
            }

        def handle_batch(self, batch):
            x, y = batch
            logits = self.model(x)

            self.batch = {
                "features": x,
                "targets": y,
                "logits": logits,
            }

    runner = CustomRunner("./logs", "cuda")
    runner.run()

Multistage run in distributed mode
------------------------------------------------

Due to multiprocessing setup during distrubuted training, the multistage runs looks a bit different:

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader, DistributedSampler
    from catalyst import dl, utils
    from catalyst.contrib.datasets import MNIST
    from catalyst.data.transforms import ToTensor


    class CustomRunner(dl.IRunner):
        def __init__(self, logdir):
            super().__init__()
            self._logdir = logdir

        def get_engine(self):
            # your could also try
            # DistributedDataParallelAMPEngine or DistributedDataParallelApexEngine engines
            return dl.DistributedDataParallelEngine()

        def get_loggers(self):
            return {
                "console": dl.ConsoleLogger(),
                "csv": dl.CSVLogger(logdir=self._logdir),
                "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
            }

        @property
        def stages(self):
            return ["train_freezed", "train_unfreezed"]

        def get_stage_len(self, stage: str) -> int:
            return 3

        def get_loaders(self, stage: str):
            # by default, Catalyst would add ``DistributedSampler`` in the framework internals
            # nevertheless, it's much easier to define this logic by yourself, isn't it?
            is_ddp = utils.get_rank() > -1
            sampler = DistributedSampler(dataset) if is_ddp else None
            loaders = {
                "train": DataLoader(
                    MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                    sampler=sampler, batch_size=32
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                    sampler=sampler, batch_size=32
                ),
            }
            return loaders

        def get_model(self, stage: str):
            # due to multiprocessing setup we have to create the model on each stage
            # to transfer the model weights between stages
            # we would use ``CheckpointCallback`` logic
            model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
            if stage == "train_freezed":  # freeze layer
                utils.set_requires_grad(model[1], False)
            else:
                utils.set_requires_grad(model, True)
            return model

        def get_criterion(self, stage: str):
            return nn.CrossEntropyLoss()

        def get_optimizer(self, stage: str, model):
            if stage == "train_freezed":
                return optim.Adam(model.parameters(), lr=1e-3)
            else:
                return optim.SGD(model.parameters(), lr=1e-1)

        def get_callbacks(self, stage: str):
            return {
                "criterion": dl.CriterionCallback(
                    metric_key="loss", input_key="logits", target_key="targets"
                ),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
                "accuracy": dl.AccuracyCallback(
                    input_key="logits", target_key="targets", topk_args=(1, 3, 5)
                ),
                "classification": dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=10
                ),
                # catalyst[ml] required
                # "confusion_matrix": dl.ConfusionMatrixCallback(
                #     input_key="logits", target_key="targets", num_classes=10
                # ),
                # the logic here is quite simple:
                # you could define which components you want to load from which checkpoints
                # by default you could load model/criterion/optimizer/scheduler components
                # and global_epoch_step/global_batch_step/global_sample_step step counters
                # from ``best`` or ``last`` checkpoints
                # for a more formal documentation, please follow CheckpointCallback docs :)
                "checkpoint": dl.CheckpointCallback(
                    self._logdir,
                    loader_key="valid",
                    metric_key="loss",
                    minimize=True,
                    save_n_best=3,
                    load_on_stage_start={
                        "model": "best",
                        "global_epoch_step": "last",
                        "global_batch_step": "last",
                        "global_sample_step": "last",
                    },
                ),
                "verbose": dl.TqdmCallback(),
            }

        def handle_batch(self, batch):
            x, y = batch
            logits = self.model(x)

            self.batch = {
                "features": x,
                "targets": y,
                "logits": logits,
            }


    if __name__ == "__main__":
        runner = CustomRunner("./logs")
        runner.run()


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
