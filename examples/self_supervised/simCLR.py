# flake8: noqa
import argparse

from sklearn.linear_model import LogisticRegression

from torch import optim

from catalyst import dl
from catalyst.contrib.losses import NTXentLoss

from src.common import add_arguments, get_contrastive_model, get_loaders
from src.datasets import DATASETS
from src.runner import SelfSupervisedRunner

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Train SimCLR")
    add_arguments(parser)
    args = parser.parse_args()

    # create model and optimizer
    model = get_contrastive_model(
        in_size=DATASETS[args.dataset]["in_size"],
        in_channels=DATASETS[args.dataset]["in_channels"],
        feature_dim=args.feature_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # define criterion
    criterion = NTXentLoss(tau=args.temperature)

    # and callbacks
    callbacks = [
        dl.CriterionCallback(
            input_key="projection_left", target_key="projection_right", metric_key="loss"
        ),
        dl.BackwardCallback(metric_key="loss"),
        dl.OptimizerCallback(metric_key="loss"),
        dl.SklearnModelCallback(
            feature_key="embedding_origin",
            target_key="target",
            train_loader="train",
            valid_loaders="valid",
            model_fn=LogisticRegression,
            predict_key="sklearn_predict",
            predict_method="predict_proba",
            C=0.1,
            solver="saga",
            max_iter=200,
        ),
        dl.ControlFlowCallbackWrapper(
            dl.AccuracyCallback(
                target_key="target", input_key="sklearn_predict", topk=(1, 3)
            ),
            loaders="valid",
        ),
    ]

    # train model
    runner = SelfSupervisedRunner()
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
