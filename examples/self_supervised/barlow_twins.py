# flake8: noqa
import argparse

from common import add_arguments, datasets, get_contrastive_model, get_loaders
from sklearn.linear_model import LogisticRegression

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from catalyst import dl
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn import BarlowTwinsLoss
from catalyst.data import SelfSupervisedDatasetWrapper

parser = argparse.ArgumentParser(description="Train Barlow Twins")
add_arguments(parser)
parser.add_argument(
    "--offdig_lambda",
    default=0.005,
    type=float,
    help="Lambda that controls the on- and off-diagonal terms",
)

if __name__ == "__main__":

    # args parse
    args = parser.parse_args()

    feature_dim, temperature = args.feature_dim, args.temperature
    offdig_lambda = args.offdig_lambda
    batch_size, epochs, num_workers = (
        args.batch_size,
        args.epochs,
        args.num_workers,
    )
    dataset = args.dataset

    # model and optimizer

    model = get_contrastive_model(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

    # criterion

    criterion = BarlowTwinsLoss(offdiag_lambda=offdig_lambda)

    callbacks = [
        dl.CriterionCallback(
            input_key="projection_left", target_key="projection_right", metric_key="loss"
        ),
        dl.SklearnModelCallback(
            feature_key="embedding_origin",
            target_key="target",
            train_loader="train",
            valid_loaders="valid",
            model_fn=LogisticRegression,
            predict_key="sklearn_predict",
            predict_method="predict_proba",
        ),
        dl.OptimizerCallback(metric_key="loss"),
        dl.ControlFlowCallback(
            dl.AccuracyCallback(
                target_key="target", input_key="sklearn_predict", topk_args=(1, 3)
            ),
            loaders="valid",
        ),
    ]

    runner = dl.SelfSupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=get_loaders(args),
        verbose=True,
        num_epochs=epochs,
        valid_loader="train",
        valid_metric="loss",
        logdir=args.logdir,
        minimize_valid_metric=True,
    )
