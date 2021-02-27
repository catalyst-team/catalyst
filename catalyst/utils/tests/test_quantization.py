import os

import numpy as np

import pytest  # noqa: F401

import torch

from catalyst.utils.quantization import quantize_model
from catalyst.contrib.nn import Flatten
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.callbacks import AccuracyCallback
from catalyst.runners import SupervisedRunner


def test_api():
    """
    Test if model can be quantize through API.
    """
    model = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 10),
    )
    q_model = quantize_model(model)
    torch.save(model.state_dict(), "model.pth")
    torch.save(q_model.state_dict(), "q_model.pth")
    model_size = os.path.getsize("model.pth")
    q_model_size = os.path.getsize("q_model.pth")
    assert q_model_size * 3.8 < model_size
    os.remove("model.pth")
    os.remove("q_model.pth")


def loader_accuracy(runner, loader):
    """Function to evaluate model.

    Args:
        runner ([type]): [description]
        loader ([type]): [description]

    Returns:
        [type]: [description]
    """
    correct, num_examples = 0, 0
    for batch in loader:
        batch = {
            "features": batch[0],
            "targets": batch[1],
        }
        logits = runner.predict_batch(batch)["logits"].detach().numpy()
        preds = logits.argmax(-1)
        num_examples += preds.shape[0]
        correct = np.equal(preds, batch["targets"]).sum()
    return correct / num_examples


def test_accuracy():
    """Test if accuracy drops too low.
    """
    model = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 10),
    )
    datasets = {
        "train": MNIST("./data", transform=ToTensor(), download=True),
        "valid": MNIST("./data", transform=ToTensor(), train=False),
    }
    dataloaders = {k: torch.utils.data.DataLoader(d, batch_size=32) for k, d in datasets.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    runner = SupervisedRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=dataloaders,
        callbacks=[AccuracyCallback(target_key="targets", input_key="logits")],
        num_epochs=4,
        criterion=torch.nn.CrossEntropyLoss(),
        valid_metric="accuracy01",
        minimize_valid_metric=False,
    )
    accuracy_before = loader_accuracy(runner, dataloaders["valid"])
    q_model = quantize_model(model)
    runner.model = q_model
    accuracy_after = loader_accuracy(runner, dataloaders["valid"])
    assert abs(accuracy_before - accuracy_after) < 0.01
