import os
from typing import Tuple, Union

import numpy as np

import torch

from catalyst.contrib.data.cv.datasets import get_image_margins
from catalyst.core import Callback, CallbackOrder, State


def pyramid_weights(height: int, width: int):
    """
    Computes a weight matrix that assigns bigger weights
    on pixels at center and less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions
    and helps dealing with prediction artifacts on tile boundaries.

    Args:
        height: Tile height
        width: Tile width

    Returns:
        Weight matrix of shape [`height` x `width`]
    """
    x_center = width * 0.5
    y_center = height * 0.5
    x_left = 0
    x_right = width
    y_top = 0
    y_bottom = height

    d_center_x = torch.pow(torch.arange(width) - x_center + 0.5, 2)
    d_center_y = torch.pow(torch.arange(height) - y_center + 0.5, 2)
    d_center = torch.sqrt(d_center_y.unsqueeze(0).transpose(0, 1) + d_center_x)

    d_edge_left = torch.pow(torch.arange(width) - x_left + 0.5, 2)
    d_edge_right = torch.pow(torch.arange(width) - x_right + 0.5, 2)
    d_edge_top = torch.pow(torch.arange(height) - y_top + 0.5, 2)
    d_edge_bottom = torch.pow(torch.arange(height) - y_bottom + 0.5, 2)

    d_edge_x = torch.sqrt(torch.min(d_edge_left, d_edge_right) + 0.5**2)
    d_edge_y = torch.sqrt(torch.min(d_edge_top, d_edge_bottom) + 0.5**2)
    d_edge = torch.min(d_edge_y.unsqueeze(0).transpose(0, 1), d_edge_x)

    frac = torch.div(d_edge, d_center + d_edge)
    alpha = (width * height) / torch.sum(frac).item()
    weights = alpha * frac

    return weights


class TiledInferenceCallback(Callback):
    """
    Callback for tiled inference, that stores resulting tensor in pre-allocated
    memory, updates its values via batch processing.
    """
    def __init__(
        self,
        save_path: str,
        image_size: Union[int, Tuple[int]],
        n_channels: int,
        tile_size: Union[int, Tuple[int]],
        tile_step: Union[int, Tuple[int]],
        output_key: str = "logits",
    ):
        """
        Args:
            save_path: file path where to save resulting array / tensor:
                file extension must be `.npy` for numpy array save
                or one of `.pt`, `.pth` for torch tensor save.
            image_size: size of large input image
            n_channels: number of channels in image
            tile_size: tile size
            tile_step: tile step
            output_key: key in batch output for obtaining neural net prediction
        """
        super().__init__(CallbackOrder.External)

        if save_path.endswith(suffix=(".pt", ".pth")):
            self.save_mode = "torch"
        elif save_path.endswith(".npy"):
            self.save_mode = "numpy"
        else:
            _, ext = os.path.splitext(save_path)
            raise ValueError(
                f"Unable to infer save mode for a file with extension {ext}"
            )

        self.save_path = save_path

        if isinstance(image_size, tuple):
            size_ndim = len(image_size)
            error_msg = (
                f"Image size must be 2-dimensional, "
                f"got {size_ndim} dimensions."
            )
            assert size_ndim == 2, error_msg
            self.image_h, self.image_w = image_size
        else:
            self.image_h, self.image_w = image_size, image_size

        if isinstance(tile_size, tuple):
            tile_size_ndim = len(tile_size)
            error_msg = (
                f"Tile size must be 2-dimensional, "
                f"got {tile_size_ndim} dimensions."
            )
            assert tile_size_ndim == 2, error_msg
            self.tile_size_h, self.tile_size_w = tile_size
        else:
            self.tile_size_h, self.tile_size_w = tile_size, tile_size

        if isinstance(tile_step, tuple):
            tile_step_ndim = len(tile_step)
            error_msg = (
                f"Tile step must be 2-dimensional, "
                f"got {tile_step_ndim} dimensions."
            )
            assert tile_step_ndim == 2, error_msg
            self.tile_step_h, self.tile_step_w = tile_step
        else:
            self.tile_step_h, self.tile_step_w = tile_step, tile_step

        margins = get_image_margins(
            self.image_h, self.image_w, self.tile_size_h, self.tile_size_w,
            self.tile_step_h, self.tile_step_w
        )

        self.margin_bottom = margins["margin_bottom"]
        self.margin_top = margins["margin_top"]
        self.margin_left = margins["margin_left"]
        self.margin_right = margins["margin_right"]

        self.n_channels = n_channels
        self.output_key = output_key

        storage_h = self.image_h + self.margin_top + self.margin_bottom
        storage_w = self.image_w + self.margin_left + self.margin_right

        self.storage = torch.zeros((n_channels, storage_h, storage_w))
        self.norm_mask = torch.zeros((1, storage_h, storage_w))
        self.weights = pyramid_weights(self.tile_size_h, self.tile_size_w)

    def on_batch_end(self, state: State):
        """
        On batch end update of resulting tensor.
        """
        self.storage = self.storage.to(state.device)
        self.norm_mask = self.norm_mask.to(state.device)
        self.weights = self.weights.to(state.device)

        items = zip(
            state.batch_out[self.output_key], state.batch_in["x"],
            state.batch_in["y"]
        )

        for predictions, x_min, y_min in items:
            x_max = x_min + self.tile_size_w
            y_max = y_min + self.tile_size_h
            crop_slice = (
                slice(None), slice(y_min, y_max), slice(x_min, x_max)
            )
            self.storage[crop_slice] += predictions * self.weights
            self.norm_mask[crop_slice] += self.weights

    def on_epoch_end(self, state: State):
        """
        On epoch end post-processing of resulting tensor, then saving it.
        """
        output = self.storage / self.norm_mask
        crop_slice = (
            slice(None), slice(self.margin_top, -self.margin_bottom),
            slice(self.margin_left, -self.margin_right)
        )
        output = output[crop_slice].cpu()
        output = output.permute(1, 2, 0)

        if self.save_mode == "torch":
            torch.save(output, self.save_path)
        elif self.save_mode == "numpy":
            np.save(self.save_path, output.numpy())
