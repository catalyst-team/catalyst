#!/usr/bin/env python
from itertools import product
import os

METHODS = ("barlow_twins", "byol", "simCLR", "supervised_contrastive")
DATASETS = ("CIFAR-10","CIFAR-100", "STL10")

BATCH_SIZE = 32

for method, dataset in product(METHODS, DATASETS):
    print(f"Start {method} on {dataset}")
    parts = ["python {method}.py", "--dataset {dataset}", "--logdir ./logs/{method}_{dataset}", "--batch_size={BATCH_SIZE}", ""]
    # os.system()