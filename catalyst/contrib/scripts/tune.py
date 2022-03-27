#!/usr/bin/env python
"""Config API and Optuna integration for AutoML hyperparameters tuning."""
from typing import Iterable
import argparse
import copy

import optuna

from catalyst import utils
from catalyst.contrib.scripts import run
from catalyst.registry import REGISTRY
from hydra_slayer import functional as F


def parse_args(args: Iterable = None, namespace: argparse.Namespace = None):
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    utils.boolean_flag(parser, "gc-after-trial", default=False)
    utils.boolean_flag(parser, "show-progress-bar", default=False)

    args, unknown_args = parser.parse_known_args(args=args, namespace=namespace)
    return vars(args), unknown_args


def main():
    """Runs the ``catalyst-tune`` script."""
    kwargs_run, unknown_args = run.parse_args()
    kwargs_tune, _ = parse_args(args=unknown_args)

    config_full = run.process_configs(**kwargs_run)
    config = copy.copy(config_full)
    config_study = config.pop("study")

    # optuna objective
    def objective(trial: optuna.trial):
        # workaround for `REGISTRY.get_from_params` - redefine `trial` var
        experiment_params, _ = F._recursive_get_from_params(
            factory_key=REGISTRY.name_key,
            get_factory_func=REGISTRY.get,
            params=config,
            shared_params={},
            var_key=REGISTRY.var_key,
            attrs_delimiter=REGISTRY.attrs_delimiter,
            vars_dict={**REGISTRY._vars_dict, "trial": trial},
        )
        runner = experiment_params["runner"]
        runner._trial = trial

        run.run_from_params(experiment_params)
        score = runner.epoch_metrics[runner._valid_loader][runner._valid_metric]

        return score

    study = REGISTRY.get_from_params(**config_study)
    study.optimize(objective, **kwargs_tune)


if __name__ == "__main__":
    main()
