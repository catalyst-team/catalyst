#!/usr/bin/env python
import argparse
from itertools import product
import os

from catalyst import utils

METHODS = ("barlow_twins", "byol", "simCLR", "supervised_contrastive")
DATASETS = ("CIFAR-10", "CIFAR-100", "STL10")
ARCH = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

BATCH_SIZE = 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run SSL {METHODS} benchmark on {DATASETS}")
    utils.boolean_flag(
        parser=parser,
        name="check",
        default=False,
        help="If this flag is on the methods will run only an epoch.",
    )
    utils.boolean_flag(parser=parser, name="verbose", default=False, help="The detailed train run loggings will be shown and saved.")
    args = parser.parse_args()

    num_epochs = 1 if args.check else 1000

    for method, dataset, arch in product(METHODS, DATASETS, ARCH):
        print(f"Start {method} on {dataset} with the model {arch}")
        cmd_parts = [
            f"python {method}.py",
            f"--dataset {dataset}",
            f"--logdir ./logs/{dataset}/{method}/{arch}",
            f"--batch-size={BATCH_SIZE}",
            f"--arch={arch}",
            f"--epochs={num_epochs}",
        ]

        if args.verbose:
            cmd_parts.append("--verbose")

        cmd = " ".join(cmd_parts)
        print(f"With a command: {cmd}")
        os.system(cmd)
