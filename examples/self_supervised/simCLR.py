# flake8: noqa
import argparse

from common import add_arguments, get_contrastive_model, get_loaders

from torch.optim import Adam

from catalyst import dl
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.dl import SelfSupervisedRunner

parser = argparse.ArgumentParser(description="Train SimCLR")
add_arguments(parser)

if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size

    # 2. model and optimizer
    model = get_contrastive_model(args.feature_dim)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # 3. criterion
    criterion = NTXentLoss(tau=args.temperature)

    callbacks = [
        dl.CriterionCallback(
            input_key="projection_left", target_key="projection_right", metric_key="loss"
        )
    ]

    runner = SelfSupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=get_loaders(args.dataset, args.batch_size, args.num_workers),
        verbose=True,
        logdir=args.logdir,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=args.epochs,
    )
