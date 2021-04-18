# based on https://github.com/pytorch/vision
import numpy as np
import torch

_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def to_tensor(pic: np.ndarray) -> torch.Tensor:
    """Convert ``numpy.ndarray`` to tensor.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        torch.Tensor: Converted image.

    Raises:
        TypeError: if `pic` is not np.ndarray
        ValueError: if `pic` is not 2/3 dimensional.
    """
    if not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be ndarray. Got {type(pic)}")
    if pic.ndim not in {2, 3}:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.ndim == 2:
        pic = pic[:, :, None]

    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    return img


def normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    Args:
        tensor: Tensor image of size (C, H, W) to be normalized.
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        torch.Tensor: Normalized Tensor image.

    Raises:
        TypeError: if `tensor` is not torch.Tensor
    """
    if not (torch.is_tensor(tensor) and tensor.ndimension() == 3):
        raise TypeError("tensor is not a torch image.")

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        """
        Args:
            transforms: list of transforms to compose.

        Example:
            >>> Compose([ToTensor(), Normalize()])
        """
        self.transforms = transforms

    def __call__(self, x):
        """Applies several transforms to the data."""
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        """@TODO: Docs. Contribution is welcome."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts numpy.ndarray (H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            torch.Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        """@TODO: Docs. Contribution is welcome."""
        return self.__class__.__name__ + "()"


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])``
    for ``n`` channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e.,
            it does not mutate the input tensor.
    """

    def __init__(self, mean, std, inplace=False):
        """
        Args:
            mean: Sequence of means for each channel.
            std: Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation in-place.
        """
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor: Tensor image of size (C, H, W) to be normalized.

        Returns:
            torch.Tensor: Normalized Tensor image.
        """
        return normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        """@TODO: Docs. Contribution is welcome."""
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


__all__ = ["Compose", "Normalize", "ToTensor", "to_tensor", "normalize"]
