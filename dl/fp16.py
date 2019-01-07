import torch
import torch.nn as nn


def copy_params(source, target):
    for i in range(len(target)):
        target[i].data.copy_(source[i].data)


def copy_grads(source, target):
    for param, param_w_grad in zip(target, source):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size())
            )
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


class Fp16Wrap(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = BN_convert_float(network.half())

    def forward(self, *args, **kwargs):
        args = list(map(lambda x: x.half(), args))
        kwargs = {key: value.half() for key, value in kwargs.items()}
        output = self.network(*args, **kwargs)
        return output
