Distributed training tutorial
==============================================================================

If you have multiple GPUs,
the most reliable way to utilize their full potential during training is to use the distributed package from PyTorch.
For such a case, there are many distributed helpers in Catalyst to make this engineering stuff a bit more user-friendly.

Please note that due to PyTorch multiprocessing realization,
GPU-based distributed training doesn't work in a notebook,
so prepare a script to run the training.
Nevertheless, XLA-based training could be run directly in the notebook.


Prepare your script
------------------------------------------------

Let's start with a simple script for ResNet9 model training on CIFAR10:

.. code-block:: python

    import os

    from torch import nn, optim
    from torch.utils.data import DataLoader

    from catalyst import dl
    from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock

    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


    def resnet9(in_channels: int, num_classes: int, size: int = 16):
        sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
        return nn.Sequential(
            conv_block(in_channels, sz),
            conv_block(sz, sz2, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
            conv_block(sz2, sz4, pool=True),
            conv_block(sz4, sz8, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
            nn.Sequential(
                nn.MaxPool2d(4), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(sz8, num_classes)
            ),
        )

    if __name__ == "__main__":
        # experiment setup
        logdir = "./logdir1"
        num_epochs = 10

        # data
        transform = Compose([
            ImageToTensor(),
            NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=transform
        )
        valid_data = CIFAR10(
            os.getcwd(), train=False, download=True, transform=transform
        )
        loaders = {
            "train": DataLoader(train_data, batch_size=32, num_workers=4),
            "valid": DataLoader(valid_data, batch_size=32, num_workers=4),
        }

        # model, criterion, optimizer, scheduler
        model = resnet9(in_channels=3, num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
        )

By default, without any additional specifications, Catalyst will utilize all available resources in
- ``DataParallel`` setup if there are several GPUs available,
- ``GPU``` setup if there is only one GPU available,
- ``CPU`` setup if there is no GPU available.


Fast DDP
------------------------------------------------

Tranks to Catalyst Python API,
you could run the same code without any change and get the distributed setup with only one line of code.
Just pass ``ddp=True`` flag during ``.train`` call:

.. code-block:: python

    import os

    from torch import nn, optim
    from torch.utils.data import DataLoader

    from catalyst import dl
    from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock

    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def resnet9(in_channels: int, num_classes: int, size: int = 16):
        sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
        return nn.Sequential(
            conv_block(in_channels, sz),
            conv_block(sz, sz2, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
            conv_block(sz2, sz4, pool=True),
            conv_block(sz4, sz8, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
            nn.Sequential(
                nn.MaxPool2d(4), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(sz8, num_classes)
            ),
        )

    if __name__ == "__main__":
        # experiment setup
        logdir = "./logdir2"
        num_epochs = 10

        # data
        transform = Compose([
            ImageToTensor(),
            NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=transform
        )
        valid_data = CIFAR10(
            os.getcwd(), train=False, download=True, transform=transform
        )
        loaders = {
            "train": DataLoader(train_data, batch_size=32, num_workers=4),
            "valid": DataLoader(valid_data, batch_size=32, num_workers=4),
        }

        # model, criterion, optimizer, scheduler
        model = resnet9(in_channels=3, num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            ddp=True,   # <-- here is the trick,
            amp=False,  # <-- here is another trick ;)
        )

Please note that you could also specify automatic mixed-precision usage with the ``amp`` flag in the same way.

In this way,
Catalyst will automatically try to make your loaders work in a distributed setup and run experiment training.

Nevertheless, it has several disadvantages,
    - without proper specification, loaders will be created again and again for each distributed worker,
    - you can't understand what is going under the hood of ``ddp=True``,
    - we can't always transfer your loaders to distributed mode correctly due to a large variety of data processing pipelines available.

For such a reason,
Catalyst API also provides a proper "low-level" API for your data preparation for the distributed setup.


DDP under the hood
------------------------------------------------

Let's create our ``CustomSupervisedRunner``
and pass the data preparation under ``CustomSupervisedRunner.get_loaders``.

.. code-block:: python

    import os

    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from catalyst import dl
    from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock

    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def resnet9(in_channels: int, num_classes: int, size: int = 16):
        sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
        return nn.Sequential(
            conv_block(in_channels, sz),
            conv_block(sz, sz2, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
            conv_block(sz2, sz4, pool=True),
            conv_block(sz4, sz8, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
            nn.Sequential(
                nn.MaxPool2d(4), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(sz8, num_classes)
            ),
        )

    class CustomSupervisedRunner(dl.SupervisedRunner):
        # here is the trick:
        def get_loaders(self, stage: str):
            transform = Compose([
                ImageToTensor(),
                NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data = CIFAR10(
                os.getcwd(), train=True, download=True, transform=transform
            )
            valid_data = CIFAR10(
                os.getcwd(), train=False, download=True, transform=transform
            )
            if self.engine.is_ddp:
                train_sampler = DistributedSampler(
                    train_data,
                    num_replicas=self.engine.world_size,
                    rank=self.engine.process_index,
                    shuffle=True,
                )
                valid_sampler = DistributedSampler(
                    valid_data,
                    num_replicas=self.engine.world_size,
                    rank=self.engine.process_index,
                    shuffle=False,
                )
            else:
                train_sampler = valid_sampler = None

            train_loader = DataLoader(
                train_data, batch_size=32, sampler=train_sampler, num_workers=4
            )
            valid_loader = DataLoader(
                valid_data, batch_size=32, sampler=train_sampler, num_workers=4
            )
            return {"train": train_loader, "valid": valid_loader}

    if __name__ == "__main__":
        # experiment setup
        logdir = "./logdir2"
        num_epochs = 10

        # model, criterion, optimizer, scheduler
        model = resnet9(in_channels=3, num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

        # model training
        runner = CustomSupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=None,  # <-- here is the trick
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            ddp=True,   # <-- now it works like a charm
            amp=False,  # <-- you can still use this trick here ;)
        )

As you can see, we have the same code,
except that the ``CustomSupervisedRunner`` now knows all the details about data preprocessing under distributed setup.
And thanks to the pure PyTorch, the code is easily readable and straightforward.


Runner under the hood
------------------------------------------------

As an extra point, you could also specify the whole experiment within ``Runner`` methods:

.. code-block:: python

    import os

    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from catalyst import dl, utils
    from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock

    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def resnet9(in_channels: int, num_classes: int, size: int = 16):
        sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
        return nn.Sequential(
            conv_block(in_channels, sz),
            conv_block(sz, sz2, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
            conv_block(sz2, sz4, pool=True),
            conv_block(sz4, sz8, pool=True),
            ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
            nn.Sequential(
                nn.MaxPool2d(4), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(sz8, num_classes)
            ),
        )

    class CustomRunner(dl.IRunner):
        def __init__(self, logdir: str):
            super().__init__()
            self._logdir = logdir

        def get_engine(self):
            return dl.DistributedDataParallelAMPEngine()

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
            transform = Compose([
                ImageToTensor(),
                NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data = CIFAR10(
                os.getcwd(), train=True, download=True, transform=transform
            )
            valid_data = CIFAR10(
                os.getcwd(), train=False, download=True, transform=transform
            )
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

            train_loader = DataLoader(
                train_data, batch_size=32, sampler=train_sampler, num_workers=4
            )
            valid_loader = DataLoader(
                valid_data, batch_size=32, sampler=train_sampler, num_workers=4
            )
            return {"train": train_loader, "valid": valid_loader}

        def get_model(self, stage: str):
            model = (
                self.model
                if self.model is not None
                else resnet9(in_channels=3, num_classes=10)
            )
            return model

        def get_criterion(self, stage: str):
            return nn.CrossEntropyLoss()

        def get_optimizer(self, stage: str, model):
            return optim.Adam(model.parameters(), lr=1e-3)

        def get_scheduler(self, stage: str, optimizer):
            return optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

        def get_callbacks(self, stage: str):
            return {
                "criterion": dl.CriterionCallback(
                    metric_key="loss", input_key="logits", target_key="targets"
                ),
                "backward": dl.BackwardCallback(metric_key="loss"),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
                "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
                "accuracy": dl.AccuracyCallback(
                    input_key="logits", target_key="targets", topk=(1, 3, 5)
                ),
                "checkpoint": dl.CheckpointCallback(
                    self._logdir,
                    loader_key="valid",
                    metric_key="accuracy",
                    minimize=False,
                    topk=1,
                ),
                # "tqdm": dl.TqdmCallback(),
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
        # experiment setup
        logdir = "./logdir3"

        runner = CustomRunner(logdir)
        runner.run()

With such low-level runner specification, you could customize every detail you want:
- hardware acceleration setup with ``get_engine``,
- data preparation with ``get_loaders``,
- experiment components with ``get_model``, ``get_optimizer``, ``get_criterion``, ``get_schduler``,
- you main training/evaluating logic withing ``handle_batch``,
- all extra components with ``get_callbacks``.


Launch your training
------------------------------------------------

In your terminal, run:

.. code-block:: bash

    python {script_name}.py

You can vary available GPUs with ``CUDA_VIBIBLE_DEVICES`` option, for example,

.. code-block:: bash

    # run only on 1st and 2nd GPUs
    CUDA_VISIBLE_DEVICES="1,2" python {script_name}.py

.. code-block:: bash

    # run only on 0, 1st and 3rd GPUs
    CUDA_VISIBLE_DEVICES="0,1,3" python {script_name}.py


What is going under the hood?
- the same model will be copied on all your available GPUs,
- then, during training, the full dataset will randomly be split between available GPUs (that will change at each epoch),
- each GPU will grab a batch (on that fractioned dataset),
- and pass it through the model, compute the loss, then back-propagate the gradients,
- then they will share their results and average them,
- since they all have the same gradients at this stage, they will all perform the same update, so the models will still be the same after the gradient step.
- then training continues with the next batch until the number of desired iterations is done.

With such specification, the distributed training is "equivalent" to training with a batch size of ```batch_size x num_gpus``
(where ``batch_size`` is what you used in your script).

During training Catalyst will automatically average all metrics
and log them on ``rank-zero`` node only. Same logic used for model checkpointing.

Resume
------------------------------------------------

During this tutorial, we have:
- review how to run distributed training with Catalyst into one single line,
- how to adapt your custom data preprocessing for the distributed training,
- and even specify the whole custom ``Runner`` if it's required.

Finally, we have reviewed the internals or distributed training and understood its "magic" under the hood.
