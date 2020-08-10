from typing import Tuple
import collections
from numbers import Number
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst.contrib.datasets import MNIST
from catalyst.contrib.dl.callbacks.gradnorm_logger import GradNormLogger
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.runner import IRunner
from catalyst.data.cv import ToTensor
from catalyst.dl import SupervisedRunner
from catalyst.registry import Model


@Model
class _SimpleNet(nn.Module):
    def __init__(self, input_shape: Tuple[int]):
        super().__init__()
        assert len(input_shape) == 3
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.flatten = nn.Flatten()

        for conv in [self.conv1, self.conv2]:
            h_kernel, w_kernel = conv.kernel_size
            h_stride, w_stride = conv.stride
            c = conv.out_channels
            h, w = self.conv2d_size_out(
                size=(h, w),
                kernel_size=(h_kernel, w_kernel),
                stride=(h_stride, w_stride),
            )

        self.fc1 = nn.Linear(in_features=c * h * w, out_features=10)

    def forward(self, x: torch.Tensor):
        for conv in [self.conv1, self.conv2]:
            x = conv(x)
            x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        return x

    @staticmethod
    def conv2d_size_out(
        *, size: Tuple[int], kernel_size: Tuple[int], stride: Tuple[int],
    ):
        """Computes output size for 2D convolution layer.
        cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
        cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
        to understand the shape for dense layer's input.

        Args:
            size (Tuple[int]): size of input.
            kernel_size (Tuple[int]): size of convolution kernel.
            stride (Tuple[int]): size of convolution stride.

        Returns:
            int: output size
        """
        size, kernel_size, stride = map(
            lambda x: torch.tensor(x, dtype=torch.int32),
            (size, kernel_size, stride),
        )
        output_size = (size - (kernel_size - 1) - 1) // stride + 1
        h, w = map(lambda x: x.item(), output_size)

        return h, w


def _get_loaders(*, root: str, batch_size: int = 1, num_workers: int = 1):
    data_transform = ToTensor()

    trainset = MNIST(
        root=root, train=True, download=True, transform=data_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers
    )
    testset = MNIST(
        root=root, train=False, download=True, transform=data_transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers
    )

    loaders = collections.OrderedDict(train=trainloader, valid=testloader)

    return loaders


class _OnBatchEndCheckGradsCallback(Callback):
    def __init__(self, prefix: str):
        super().__init__(CallbackOrder.external)
        self.prefix = prefix

    def on_batch_end(self, runner: IRunner):
        if not runner.is_train_loader:
            return

        for layer in ["conv1", "conv2", "fc1"]:
            for weights in ["weight", "bias"]:
                tag = f"{self.prefix}/{layer}/{weights}"
                assert tag in runner.batch_metrics
                assert isinstance(runner.batch_metrics[tag], Number)

        tag = f"{self.prefix}/total"
        assert tag in runner.batch_metrics
        assert isinstance(runner.batch_metrics[tag], Number)


def test_save_model_grads():
    """
    Tests a feature of `OptimizerCallback` for saving model gradients
    """
    logdir = "./logs"
    dataset_root = "./data"
    loaders = _get_loaders(root=dataset_root, batch_size=4, num_workers=1)
    images, _ = next(iter(loaders["train"]))
    _, c, h, w = images.shape
    input_shape = (c, h, w)

    model = _SimpleNet(input_shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    criterion_callback = CriterionCallback()
    optimizer_callback = OptimizerCallback()
    save_model_grads_callback = GradNormLogger()
    prefix = save_model_grads_callback.grad_norm_prefix
    test_callback = _OnBatchEndCheckGradsCallback(prefix)

    callbacks = collections.OrderedDict(
        loss=criterion_callback,
        optimizer=optimizer_callback,
        grad_norm=save_model_grads_callback,
        test_callback=test_callback,
    )

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        callbacks=callbacks,
        check=True,
        verbose=True,
    )

    shutil.rmtree(logdir)
    # shutil.rmtree(dataset_root)
