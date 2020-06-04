from typing import List, Tuple, Union
import os
from pathlib import Path

import imageio
import numpy as np
from skimage.color import label2rgb

import torch
import torch.nn.functional as F

from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl import utils
from catalyst.contrib.data.cv.datasets import get_image_margins


def pyramid_weights(height: int, width: int) -> torch.Tensor:
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

    d_edge_x = torch.sqrt(torch.min(d_edge_left, d_edge_right) + 0.5 ** 2)
    d_edge_y = torch.sqrt(torch.min(d_edge_top, d_edge_bottom) + 0.5 ** 2)
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
        save_dir: str,
        image_size: Union[int, Tuple[int], List[int]],
        num_classes: int,
        tile_size: Union[int, Tuple[int], List[int]],
        tile_step: Union[int, Tuple[int], List[int]],
        threshold: float = 0.5,
        output_key: str = "logits",
        mask_key: str = "mask",
        class_first: bool = True,
    ):
        """
        Args:
            save_dir: directory where resulting objects are stored.
            image_size: size of large input image.
            num_classes: number of channels in output mask.
            tile_size: tile size.
            tile_step: tile step.
            threshold: threshold for masking
            output_key: key in batch output for obtaining neural net
                prediction.
            mask_key: key for naming a file with output mask.
            class_first: if True, then saved array will have
                dimensions order (num_classes, height, width),
                if False, then (height, width, num_classes).
        """
        super().__init__(CallbackOrder.External)

        assert num_classes >= 1, (
            f"Number of classes must be greater or equal to 1, "
            f"got {num_classes}."
        )

        self.save_dir = Path(save_dir)

        if isinstance(image_size, (tuple, list)):
            size_ndim = len(image_size)
            error_msg = (
                f"Image size must be 2-dimensional, "
                f"got {size_ndim} dimensions."
            )
            assert size_ndim == 2, error_msg
            self.image_h, self.image_w = image_size
        else:
            self.image_h, self.image_w = image_size, image_size

        if isinstance(tile_size, (tuple, list)):
            tile_size_ndim = len(tile_size)
            error_msg = (
                f"Tile size must be 2-dimensional, "
                f"got {tile_size_ndim} dimensions."
            )
            assert tile_size_ndim == 2, error_msg
            self.tile_size_h, self.tile_size_w = tile_size
        else:
            self.tile_size_h, self.tile_size_w = tile_size, tile_size

        if isinstance(tile_step, (tuple, list)):
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
            self.image_h,
            self.image_w,
            self.tile_size_h,
            self.tile_size_w,
            self.tile_step_h,
            self.tile_step_w,
        )

        self.margin_bottom = margins["margin_bottom"]
        self.margin_top = margins["margin_top"]
        self.margin_left = margins["margin_left"]
        self.margin_right = margins["margin_right"]

        self.num_classes = num_classes
        self.output_key = output_key
        self.mask_key = mask_key
        self.class_first = class_first

        storage_h = self.image_h + self.margin_top + self.margin_bottom
        storage_w = self.image_w + self.margin_left + self.margin_right

        self.storage = torch.zeros((num_classes, storage_h, storage_w))
        self.norm_mask = torch.zeros((1, storage_h, storage_w))
        self.weights = pyramid_weights(self.tile_size_h, self.tile_size_w)

        self.threshold = threshold

    def on_batch_end(self, state: State):
        """
        On batch end update of resulting tensor.
        """
        self.storage = self.storage.to(state.device)
        self.norm_mask = self.norm_mask.to(state.device)
        self.weights = self.weights.to(state.device)

        output = state.batch_out[self.output_key]

        for predictions, x_min, y_min in zip(
            output, state.batch_in["x"], state.batch_in["y"],
        ):
            x_max = x_min + self.tile_size_w
            y_max = y_min + self.tile_size_h
            crop_slice = (
                slice(None),
                slice(y_min, y_max),
                slice(x_min, x_max),
            )
            self.storage[crop_slice] += predictions * self.weights
            self.norm_mask[crop_slice] += self.weights

    def on_epoch_end(self, state: State):
        """
        On epoch end post-processing of resulting tensor, then saving it.
        """
        eps = torch.finfo(self.norm_mask.dtype).eps
        self.norm_mask = torch.clamp_min(self.norm_mask, eps)
        logits = self.storage / self.norm_mask
        _, h, w = logits.shape
        crop_slice = (
            slice(None),
            slice(self.margin_top, h - self.margin_bottom),
            slice(self.margin_left, w - self.margin_right),
        )
        logits = logits[crop_slice]

        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=0)

        mask = torch.zeros_like(probs[0], dtype=torch.int32)

        for i, channel in enumerate(probs):
            mask[channel >= self.threshold] = i + 1

        if not self.class_first:
            probs = probs.permute(1, 2, 0)
            mask = mask.permute(1, 2, 0)

        probs = probs.cpu().numpy()
        mask = mask.cpu().numpy()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.save_dir / f"probs.npy", probs)
        np.save(self.save_dir / f"mask.npy", mask)


class InferMaskCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        out_dir=None,
        out_prefix=None,
        input_key=None,
        output_key=None,
        name_key=None,
        mean=None,
        std=None,
        threshold: float = 0.5,
        mask_strength: float = 0.5,
        mask_type: str = "soft",
    ):
        """
        Args:
            @TODO: Docs. Contribution is welcome
        """
        super().__init__(CallbackOrder.Internal)
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.mean = mean or np.array([0.485, 0.456, 0.406])
        self.std = std or np.array([0.229, 0.224, 0.225])
        assert input_key is not None
        assert output_key is not None
        self.threshold = threshold
        self.mask_strength = mask_strength
        self.mask_type = mask_type
        self.input_key = input_key
        self.output_key = output_key
        self.name_key = name_key
        self.counter = 0
        self._keys_from_state = ["out_dir", "out_prefix"]

    def on_stage_start(self, state: State):
        """Stage start hook.

        Args:
            state (State): current state
        """
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        self.out_prefix = (
            self.out_prefix if self.out_prefix is not None else ""
        )
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, state: State):
        """Loader start hook.

        Args:
            state (State): current state
        """
        lm = state.loader_name
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, state: State):
        """Batch end hook.

        Args:
            state (State): current state
        """
        lm = state.loader_name
        names = state.input.get(self.name_key, [])

        features = state.input[self.input_key].detach().cpu()
        images = utils.tensor_to_ndimage(features)

        logits = state.output[self.output_key]
        logits = (
            torch.unsqueeze_(logits, dim=1)
            if len(logits.shape) < 4
            else logits
        )

        if self.mask_type == "soft":
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.detach().cpu().numpy()

        masks = []
        for probability in probabilities:
            mask = np.zeros_like(probability[0], dtype=np.int32)
            for i, ch in enumerate(probability):
                mask[ch >= self.threshold] = i + 1
            masks.append(mask)

        for i, (image, mask) in enumerate(zip(images, masks)):
            try:
                suffix = names[i]
            except IndexError:
                suffix = f"{self.counter:06d}"
            self.counter += 1

            mask = label2rgb(mask, bg_label=0)

            image = (
                image * (1 - self.mask_strength) + mask * self.mask_strength
            )
            image = (image * 255).clip(0, 255).round().astype(np.uint8)

            filename = f"{self.out_prefix}/{lm}/{suffix}.jpg"
            imageio.imwrite(filename, image)


__all__ = ["InferMaskCallback"]
