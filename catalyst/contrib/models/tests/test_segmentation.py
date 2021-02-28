# flake8: noqa
import torch


def check_unet(net_fn):
    net = net_fn()
    # print(net)
    # print("-"*80)
    in_tensor = torch.Tensor(4, 3, 256, 256)
    out_tensor_true = torch.Tensor(4, 1, 256, 256)
    # print(in_tensor.shape)
    # print("-"*80)
    out_tensor = net(in_tensor)
    # print(out_tensor.shape)
    # print("-"*80)
    # print(sum(p.numel() for p in net.parameters()))
    assert out_tensor.shape == out_tensor_true.shape, f"{net_fn} feels bad"
    print(f"{net_fn} feels good")
    return net


def test_unet():
    from catalyst.contrib.models.cv import Unet

    net = check_unet(Unet)


def test_linknet():
    from catalyst.contrib.models.cv import Linknet

    net = check_unet(Linknet)


def test_fpnnet():
    from catalyst.contrib.models.cv import FPNUnet

    net = check_unet(FPNUnet)


def test_pspnet():
    from catalyst.contrib.models.cv import PSPnet

    net = check_unet(PSPnet)


def test_unet2():
    from catalyst.contrib.models.cv import ResnetUnet

    net = check_unet(ResnetUnet)


def test_linknet2():
    from catalyst.contrib.models.cv import ResnetLinknet

    net = check_unet(ResnetLinknet)


def test_fpnnet2():
    from catalyst.contrib.models.cv import ResnetFPNUnet

    net = check_unet(ResnetFPNUnet)


def test_pspnet2():

    from catalyst.contrib.models.cv import ResnetPSPnet

    net = check_unet(ResnetPSPnet)
