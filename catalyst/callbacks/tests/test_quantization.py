# flake8: noqa
import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def test_pruning_callback() -> None:
    """Quantize model"""
    loaders = {
        "train": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32,
        ),
        "valid": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32,
        ),
    }
    model = nn.Sequential(Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        callbacks=[dl.QuantizationCallback(logdir="./logs")],
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        logdir="./logs",
        check=True,
    )
    assert os.path.isfile("./logs/quantized.pth")
