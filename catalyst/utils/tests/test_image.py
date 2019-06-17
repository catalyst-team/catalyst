import numpy as np
import torch
from torchvision.transforms.functional import normalize, to_tensor

from catalyst.utils.image import tensor_to_ndimage, \
    _IMAGENET_MEAN, _IMAGENET_STD


def test_tensor_to_ndimage():
    orig_images = np.random.randint(0, 255, (2, 20, 10, 3), np.uint8)

    torch_images = torch.stack([
        normalize(to_tensor(im), _IMAGENET_MEAN, _IMAGENET_STD)
        for im in orig_images
    ], dim=0)

    byte_images = tensor_to_ndimage(torch_images, dtype=np.uint8)
    float_images = tensor_to_ndimage(torch_images, dtype=np.float32)

    assert np.allclose(byte_images, orig_images)
    assert np.allclose(float_images, orig_images / 255, atol=1e-3, rtol=1e-3)

    assert np.allclose(
        tensor_to_ndimage(torch_images[0]),
        orig_images[0] / 255,
        atol=1e-3, rtol=1e-3
    )
