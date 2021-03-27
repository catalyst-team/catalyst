import numpy as np
import torch

# def format_metric(name: str, value: float) -> str:
#     """Format metric.
#
#     Metric will be returned in the scientific format if 4
#     decimal chars are not enough (metric value lower than 1e-4).
#
#     Args:
#         name: metric name
#         value: value of metric
#
#     Returns:
#         str: formatted metric
#     """
#     if value < 1e-4:
#         return f"{name}={value:1.3e}"
#     return f"{name}={value:.4f}"


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Creates tensor from RGB image.

    Args:
        image: RGB image stored as np.ndarray

    Returns:
        tensor
    """
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image
