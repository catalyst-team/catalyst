# flake8: noqa
import argparse

from common import add_arguments, get_contrastive_model, get_loaders
from sklearn.linear_model import LogisticRegression

import torch
from torch.optim import Adam

from catalyst import dl
from catalyst.contrib.losses import SupervisedContrastiveLoss


def concat(*tensors):
    return torch.cat(tensors)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Train Supervised Contrastive")
    add_arguments(parser)
    args = parser.parse_args()

    # create model and optimizer
    model = get_contrastive_model(args.feature_dim, args.arch, args.frozen)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # define criterion with triplets sampling
    criterion = SupervisedContrastiveLoss(tau=args.temperature)

    # and callbacks
    callbacks = [
        dl.BatchTransformCallback(
            input_key=["projection_left", "projection_right"],
            output_key="full_projection",
            scope="on_batch_end",
            transform=concat,
        ),
        dl.BatchTransformCallback(
            input_key=["target", "target"],
            output_key="full_targets",
            scope="on_batch_end",
            transform=concat,
        ),
        dl.CriterionCallback(
            input_key="full_projection", target_key="full_targets", metric_key="loss"
        ),
        dl.SklearnModelCallback(
            feature_key="full_projection",
            target_key="full_targets",
            train_loader="train",
            valid_loaders="valid",
            model_fn=LogisticRegression,
            predict_key="sklearn_predict",
            predict_method="predict_proba",
            C=0.1,
            solver="saga",
            max_iter=200,
        ),
    ]

    # train model
    runner = dl.SelfSupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=get_loaders(args.dataset, args.batch_size, args.num_workers),
        num_epochs=args.epochs,
        logdir=args.logdir,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=args.verbose,
        # check=args.check,
    )
