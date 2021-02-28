# flake8: noqa
# from typing import Dict, Tuple, Union
# import collections
# from pathlib import Path
# import shutil
#
# import torch
# from torch import nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader
#
# from catalyst.callbacks import CriterionCallback, OptimizerCallback, TracingCallback
# from catalyst.data.transforms import ToTensor
# from catalyst.contrib.datasets import MNIST
# from catalyst.core.callback import Callback, CallbackOrder
# from catalyst.core.runner import IRunner
# from catalyst.registry import REGISTRY
# from catalyst.runners.supervised import SupervisedRunner
# from catalyst.utils import get_device, get_trace_name
#
#
# @REGISTRY.add
# class _TracedNet(nn.Module):
#     """
#     Simple model for the testing.
#     """
#
#     def __init__(self, input_shape: Tuple[int]):
#         """
#         Args:
#             input_shape: Shape of input tensor.
#         """
#         super().__init__()
#         assert len(input_shape) == 3
#         c, h, w = input_shape
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
#         self.flatten = nn.Flatten()
#
#         for conv in [self.conv1, self.conv2]:
#             h_kernel, w_kernel = conv.kernel_size
#             h_stride, w_stride = conv.stride
#             c = conv.out_channels
#             h, w = self.conv2d_size_out(
#                 size=(h, w), kernel_size=(h_kernel, w_kernel), stride=(h_stride, w_stride),
#             )
#
#         self.fc1 = nn.Linear(in_features=c * h * w, out_features=10)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input tensor
#
#         Returns:
#             torch.Tensor: Output tensor
#         """
#         for conv in [self.conv1, self.conv2]:
#             x = conv(x)
#             x = self.relu(x)
#
#         x = self.flatten(x)
#         x = self.fc1(x)
#         return x
#
#     @staticmethod
#     def conv2d_size_out(
#         *, size: Tuple[int], kernel_size: Tuple[int], stride: Tuple[int],
#     ) -> Tuple[int, int]:
#         """
#         Computes output size for 2D convolution layer.
#         cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
#         cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
#         to understand the shape for dense layer's input.
#
#         Args:
#             size: size of input.
#             kernel_size: size of convolution kernel.
#             stride: size of convolution stride.
#
#         Returns:
#             Tuple[int, int]: output size
#         """
#         size, kernel_size, stride = map(
#             lambda x: torch.tensor(x, dtype=torch.int32), (size, kernel_size, stride),
#         )
#         output_size = (size - (kernel_size - 1) - 1) // stride + 1
#         h, w = map(lambda x: x.item(), output_size)
#
#         return h, w
#
#
# def _get_loaders(*, root: str, batch_size: int = 1, num_workers: int = 1) -> Dict[str, DataLoader]:
#     """
#     Function to get loaders just for testing.
#
#     Args:
#         root: Path to root of dataset.
#         batch_size: Batch size.
#         num_workers: Num of workers.
#
#     Returns:
#         Dict[str, DataLoader]: Dict of loaders.
#     """
#     data_transform = ToTensor()
#
#     trainset = MNIST(root=root, train=True, download=True, transform=data_transform)
#     trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
#     testset = MNIST(root=root, train=False, download=True, transform=data_transform)
#     testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
#
#     loaders = collections.OrderedDict(train=trainloader, valid=testloader)
#
#     return loaders
#
#
# class _OnStageEndCheckModelTracedCallback(Callback):
#     """
#     Callback to test traced model at the end of the stage.
#     """
#
#     def __init__(self, path: Union[str, Path], inputs: torch.Tensor):
#         """
#         Args:
#             path (Union[str, Path]): Path to traced model.
#             inputs: Input samples.
#         """
#         super().__init__(CallbackOrder.external)
#         self.path: Path = Path(path)
#         self.inputs: torch.Tensor = inputs
#         self.device = get_device()
#
#     def on_stage_end(self, runner: "IRunner"):
#         """
#         Args:
#             runner: current runner
#         """
#         assert self.path.exists(), "Traced model was not found"
#
#         traced_model = torch.jit.load(str(self.path))
#         traced_model = traced_model.to(self.device)
#         self.inputs = self.inputs.to(self.device)
#         result = traced_model(self.inputs)
#
#         assert result is not None and isinstance(
#             result, torch.Tensor
#         ), "Traced model is not working correctly"
#
#
# def test_tracer_callback():
#     """
#     Tests a feature of `TracingCallback` for model tracing during training
#     """
#     logdir = "./logs"
#     dataset_root = "./data"
#     loaders = _get_loaders(root=dataset_root, batch_size=4, num_workers=1)
#     images, targets = next(iter(loaders["train"]))
#     _, c, h, w = images.shape
#     input_shape = (c, h, w)
#
#     model = _TracedNet(input_shape)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters())
#
#     method_name = "forward"
#     mode = "eval"
#     requires_grad = False
#     checkpoint_name = "best"
#     opt_level = None
#
#     trace_name = get_trace_name(
#         method_name=method_name,
#         mode=mode,
#         requires_grad=requires_grad,
#         additional_string=checkpoint_name,
#     )
#     tracing_path = Path(logdir) / "trace" / trace_name
#     criterion_callback = CriterionCallback()
#     optimizer_callback = OptimizerCallback()
#     tracer_callback = TracingCallback(
#         metric="loss",
#         minimize=False,
#         trace_mode=mode,
#         mode=checkpoint_name,
#         do_once=True,
#         method_name=method_name,
#         requires_grad=requires_grad,
#         opt_level=opt_level,
#     )
#     test_callback = _OnStageEndCheckModelTracedCallback(path=tracing_path, inputs=images)
#
#     callbacks = collections.OrderedDict(
#         loss=criterion_callback,
#         optimizer=optimizer_callback,
#         tracer_callback=tracer_callback,
#         test_callback=test_callback,
#     )
#
#     runner = SupervisedRunner(input_key="x")
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         loaders=loaders,
#         logdir=logdir,
#         callbacks=callbacks,
#         check=True,
#         verbose=True,
#     )
#
#     shutil.rmtree(logdir)
#     # shutil.rmtree(dataset_root)
