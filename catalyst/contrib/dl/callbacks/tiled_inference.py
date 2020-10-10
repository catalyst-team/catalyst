from collections import namedtuple
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F  # noqa: N812, WPS301

from catalyst.contrib.data.cv.datasets.image import Edges, Shape
from catalyst.core import Callback, CallbackOrder, State

Allocated = namedtuple("Allocated", ["storage", "norm_mask", "weights"])


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
        *,
        save_dir: str,
        threshold: float = 0.5,
        output_key: str = "logits",
        mask_key: str = "mask",
        class_first: bool = True,
    ):
        """
        Args:
            save_dir: directory where resulting objects are stored.
            threshold: threshold for masking
            output_key: key in batch output for obtaining neural net
                prediction.
            mask_key: key for naming a file with output mask.
            class_first: if True, then saved array will have
                dimensions order (num_classes, height, width),
                if False, then (height, width, num_classes).
        """
        super().__init__(CallbackOrder.External)

        self.save_dir = Path(save_dir)
        self.output_key = output_key
        self.mask_key = mask_key
        self.class_first = class_first
        self.threshold = threshold

        self.image_idx = []
        self.allocated = {}
        self.margins = {}
        self.probs = None
        self.masks = None

        self.probs_path = self.save_dir / f"probs.npy"
        self.masks_path = self.save_dir / f"masks.npy"

    def on_stage_start(self, state: State):
        """
        On stage start hook.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.probs_path.exists():
            self.probs_path.unlink()
        if self.masks_path.exists():
            self.masks_path.unlink()

    def on_loader_start(self, state: State):
        """
        On loader start initialization.
        """
        self.image_idx = []
        self.allocated = {}
        self.margins = {}
        self.probs = None
        self.masks = None

    def on_batch_end(self, state: State):
        """
        On batch end update of resulting tensor.
        """
        output = state.output[self.output_key]

        for (
            predictions,
            idx,
            x_min,
            y_min,
            storage_size,
            margins,
            tile_size,
        ) in zip(
            output,
            state.input["id"],
            state.input["x"],
            state.input["y"],
            state.input["storage_size"],
            state.input["margins"],
            state.input["tile_size"],
        ):
            idx = idx.item()
            x_min = x_min.item()
            y_min = y_min.item()
            storage_size = storage_size.cpu().numpy().tolist()
            margins = margins.cpu().numpy().tolist()
            tile_size = tile_size.cpu().numpy().tolist()

            num_classes, *_ = predictions.shape
            storage_size = Shape(*storage_size)
            margins = Edges(*margins)
            tile_size = Shape(*tile_size)

            if idx not in self.allocated:
                self.image_idx.append(idx)

                storage = torch.zeros(
                    (num_classes, storage_size.height, storage_size.width)
                )
                norm_mask = torch.zeros(
                    (1, storage_size.height, storage_size.width)
                )
                weights = pyramid_weights(tile_size.height, tile_size.width)
                self.allocated[idx] = Allocated(
                    storage=storage, norm_mask=norm_mask, weights=weights,
                )

                self.margins[idx] = margins

            x_max = x_min + tile_size.width
            y_max = y_min + tile_size.height
            crop_slice = (
                slice(None),
                slice(y_min, y_max),
                slice(x_min, x_max),
            )

            storage = self.allocated[idx].storage.to(state.device)
            norm_mask = self.allocated[idx].norm_mask.to(state.device)
            weights = self.allocated[idx].weights.to(state.device)

            storage[crop_slice] += predictions * weights
            norm_mask[crop_slice] += weights

            self.allocated[idx] = Allocated(
                storage=storage, norm_mask=norm_mask, weights=weights,
            )

            if (norm_mask != 0.0).all().item():
                self._store_ready(idx)
                self.allocated.pop(idx)

    def _store_ready(self, idx: int):
        allocated = self.allocated[idx]
        margins = self.margins[idx]
        eps = torch.finfo(allocated.norm_mask.dtype).eps
        norm_mask = torch.clamp_min(allocated.norm_mask, eps)
        logits = allocated.storage / norm_mask
        num_classes, h, w = logits.shape
        crop_slice = (
            slice(None),
            slice(margins.top, h - margins.bottom),
            slice(margins.left, w - margins.right),
        )
        logits = logits[crop_slice]

        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=0)

        masks = torch.zeros_like(probs[0], dtype=torch.int32)

        for i, channel in enumerate(probs):
            masks[channel >= self.threshold] = i + 1

        if not self.class_first:
            probs = probs.permute(1, 2, 0)
            masks = masks.permute(1, 2, 0)

        probs = probs.detach().cpu().unsqueeze(dim=0).numpy()
        masks = masks.detach().cpu().unsqueeze(dim=0).numpy()

        if self.probs is None:
            self.probs = probs
        else:
            self.probs = np.vstack([self.probs, probs])

        if self.masks is None:
            self.masks = masks
        else:
            self.masks = np.vstack([self.masks, masks])

    def on_loader_end(self, state: State):
        """
        On loader end post-processing of resulting tensor, then saving it.
        """
        np.save(self.probs_path, self.probs)
        np.save(self.masks_path, self.masks)


__all__ = ["pyramid_weights", "TiledInferenceCallback"]
