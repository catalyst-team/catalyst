# flake8: noqa
from tempfile import TemporaryDirectory

from pytest import mark

from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.datasets import MnistMLDataset, MnistQGDataset
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.contrib.models import MnistSimpleNet
from catalyst.data.sampler import BatchBalanceClassSampler
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import DATA_ROOT


class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
            }
        else:
            images, targets, is_query = (
                batch["features"].float(),
                batch["targets"].long(),
                batch["is_query"].bool(),
            )
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "is_query": is_query,
            }


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:

        # 1. train and valid loaders
        train_dataset = MnistMLDataset(root=DATA_ROOT)
        sampler = BatchBalanceClassSampler(
            labels=train_dataset.get_labels(),
            num_classes=5,
            num_samples=10,
            num_batches=10,
        )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=sampler)

        valid_dataset = MnistQGDataset(root=DATA_ROOT, gallery_fraq=0.2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

        # 2. model and optimizer
        model = MnistSimpleNet(out_features=16)
        optimizer = Adam(model.parameters(), lr=0.001)

        # 3. criterion with triplets sampling
        sampler_inbatch = HardTripletsSampler(norm_required=False)
        criterion = TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch
        )

        # 4. training with catalyst Runner
        callbacks = [
            dl.ControlFlowCallbackWrapper(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.ControlFlowCallbackWrapper(
                dl.CMCScoreCallback(
                    embeddings_key="embeddings",
                    labels_key="targets",
                    is_query_key="is_query",
                    topk=[1],
                ),
                loaders="valid",
            ),
            dl.PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="cmc01",
                minimize=False,
                valid=2,
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": valid_loader},
            verbose=False,
            logdir=logdir,
            valid_loader="valid",
            valid_metric="cmc01",
            minimize_valid_metric=False,
            num_epochs=2,
        )


# Torch
def test_classification_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_classification_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_cuda1():
#     train_experiment("cuda:1")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
)
def test_classification_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found"
)
def test_classification_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_classification_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
