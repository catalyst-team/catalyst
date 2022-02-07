# flake8: noqa
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl, utils
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS

if SETTINGS.ml_required:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

TRAIN_EPOCH = 5
LR = 0.001
RANDOM_STATE = 42


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        utils.set_global_seed(RANDOM_STATE)
        # 1. generate data
        num_samples, num_features, num_classes = int(1e4), int(30), 3
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features,
            n_repeated=0,
            n_redundant=0,
            n_classes=num_classes,
            n_clusters_per_class=1,
        )
        X, y = torch.tensor(X), torch.tensor(y)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, num_workers=1, shuffle=True)

        # 2. model, optimizer and scheduler
        hidden_size, out_features = 20, 16
        model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # 3. criterion with triplets sampling
        sampler_inbatch = HardTripletsSampler(norm_required=False)
        criterion = TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch
        )

        # 4. training with catalyst Runner
        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch) -> None:
                features, targets = batch["features"].float(), batch["targets"].long()
                embeddings = self.model(features)
                self.batch = {
                    "embeddings": embeddings,
                    "targets": targets,
                }

        callbacks = [
            dl.SklearnModelCallback(
                feature_key="embeddings",
                target_key="targets",
                train_loader="train",
                valid_loaders="valid",
                model_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                random_state=RANDOM_STATE,
                n_estimators=100,
            ),
            dl.ControlFlowCallbackWrapper(
                dl.AccuracyCallback(
                    target_key="targets", input_key="sklearn_predict", topk=(1, 3)
                ),
                loaders="valid",
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            scheduler=scheduler,
            loaders={"train": loader, "valid": loader},
            verbose=False,
            valid_loader="valid",
            valid_metric="accuracy01",
            minimize_valid_metric=False,
            num_epochs=TRAIN_EPOCH,
            logdir=logdir,
        )

        best_accuracy = max(
            epoch_metrics["valid"]["accuracy01"]
            for epoch_metrics in runner.experiment_metrics.values()
        )

        assert best_accuracy > 0.9


requirements_satisfied = SETTINGS.ml_required


# Torch
@mark.skipif(not requirements_satisfied, reason="catalyst[ml] and catalyst[cv] required")
def test_run_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(
    not all([requirements_satisfied, IS_CUDA_AVAILABLE]),
    reason="CUDA device is not available",
)
def test_run_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_run_on_torch_cuda1():
#     train_experiment("cuda:1")


@mark.skipif(
    not all([requirements_satisfied, (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2)]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not all([requirements_satisfied, (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2)]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not all([requirements_satisfied, (IS_CUDA_AVAILABLE and SETTINGS.amp_required)]),
    reason="No CUDA or AMP found",
)
def test_run_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not all(
        [
            requirements_satisfied,
            (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_run_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not all(
#         [
#             requirements_satisfied,
#             (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
