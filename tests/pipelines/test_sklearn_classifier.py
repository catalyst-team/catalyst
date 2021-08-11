# flake8: noqa
from catalyst import utils

utils.set_global_seed(42)
import os

from pytest import mark
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import data, dl
from catalyst.contrib import datasets, models, nn
from catalyst.data.transforms import Compose, Normalize, ToTensor
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

TRAIN_EPOCH = 20
LR = 0.0001


def train_experiment(device, engine=None):
    # 1. train, valid and test loaders
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MnistMLDataset(root=os.getcwd(), download=True, transform=transforms)
    sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
    train_loader = DataLoader(
        dataset=train_dataset, sampler=sampler, batch_size=sampler.batch_size
    )

    valid_dataset = datasets.MNIST(
        root=os.getcwd(), transform=transforms, train=True, download=True
    )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

    test_dataset = datasets.MNIST(
        root=os.getcwd(), transform=transforms, train=False, download=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024)

    # 2. model and optimizer
    model = models.MnistSimpleNet(out_features=32, normalize=True)
    optimizer = Adam(model.parameters(), lr=LR)

    # 3. criterion with triplets sampling
    sampler_inbatch = data.HardTripletsSampler(norm_required=False)
    criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

    # 4. training with catalyst Runner
    class CustomRunner(dl.SupervisedRunner):
        def handle_batch(self, batch) -> None:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
            }

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(input_key="embeddings", target_key="targets", metric_key="loss"),
            loaders="train",
        ),
        dl.ControlFlowCallback(
            dl.SklearnModelCallback(
                feature_key="embeddings",
                target_key="targets",
                train_loader="valid",
                valid_loader="infer",
                sklearn_classifier_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                n_estimators=100,
            ),
            filter_fn=lambda s, e, l: e > TRAIN_EPOCH,
        ),
        dl.ControlFlowCallback(
            dl.AccuracyCallback(
                target_key="targets", input_key="sklearn_predict", topk_args=(1, 3)
            ),
            filter_fn=lambda s, e, l: l == "infer" and e > TRAIN_EPOCH,
        ),
    ]

    runner = CustomRunner(input_key="features", output_key="embeddings")
    runner.train(
        engine=engine or dl.DeviceEngine(device),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": valid_loader, "infer": test_loader},
        verbose=False,
        logdir="./logs",
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=TRAIN_EPOCH + 1,
    )
    print(runner.loader_metrics["accuracy"].item())
    assert runner.loader_metrics["accuracy"].item() > 0.7


@mark.skipif(not SETTINGS.ml_required, reason="catalyst[ml] required")
def test_on_cpu():
    train_experiment("cpu")
