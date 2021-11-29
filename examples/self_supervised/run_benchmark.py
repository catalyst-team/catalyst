#!/usr/bin/env python
import argparse
from itertools import product
import os

METHODS = ("barlow_twins", "byol", "simCLR", "supervised_contrastive")
DATASETS = ("CIFAR-10", "CIFAR-100", "STL10")

BATCH_SIZE = 32
checks = False

parser = argparse.ArgumentParser(description=f"Run SSL {METHODS} benchmark on {DATASETS}")
parser.add_argument(
    "--check",
    dest="check",
    action="store_true",
    help=(
        "If this flag use the methods will run only on few batches"
        "(quick test that everything working)."
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()

    for method, dataset in product(METHODS, DATASETS):
        print(f"Start {method} on {dataset}")
        cmd_parts = [
            f"python {method}.py",
            f"--dataset {dataset}",
            f"--logdir ./logs/{method}_{dataset}",
            f"--batch_size={BATCH_SIZE}",
        ]

        if args.check:
            cmd_parts.append("--check")

        cmd = " ".join(cmd_parts)
        print(f"With a command: {cmd}")
        os.system(cmd)
