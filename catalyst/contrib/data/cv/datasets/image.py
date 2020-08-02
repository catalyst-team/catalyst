from typing import Callable, List, Optional, Tuple, Union
from bisect import bisect_right
from collections import namedtuple, OrderedDict
import math
from pathlib import Path

import cv2
import numpy as np
import skimage.io

from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
import torch
from torch.utils.data import Dataset

Edges = namedtuple("Edges", ["bottom", "top", "left", "right"])
Shape = namedtuple("Shape", ["height", "width"])


def get_image_margins(
    image_h: int,
    image_w: int,
    tile_size_h: int,
    tile_size_w: int,
    tile_step_h: int,
    tile_step_w: int,
) -> Edges:
    """
    Compute image margins to fit tile parameters.

    Args:
        image_h: image height
        image_w: image width
        tile_size_h: tile size in height
        tile_size_w: tile size in width
        tile_step_h: tile step in height
        tile_step_w: tile step in width

    Returns:
        OrderedDict with margins at all 4 edges of the image.
    """
    assert tile_size_h <= image_h, (
        f"`tile_size_h` ({tile_size_h}) must be less or equal than "
        f"`image_h` ({image_h})."
    )
    assert tile_size_w <= image_w, (
        f"`tile_size_w` ({tile_size_w}) must be less or equal than "
        f"`image_w` ({image_w})."
    )
    assert tile_size_h > tile_step_h, (
        f"`tile_size_h` ({tile_size_h}) must be greater than "
        f"`tile_step_h` ({tile_step_h})."
    )
    assert tile_size_w > tile_step_w, (
        f"`tile_size_w` ({tile_size_w}) must be greater than "
        f"`tile_step_w` ({tile_step_w})."
    )

    overlap_h = tile_size_h - tile_step_h
    overlap_w = tile_size_w - tile_step_w

    num_tiles_h = math.ceil((image_h - overlap_h) / tile_step_h)
    num_tiles_w = math.ceil((image_w - overlap_w) / tile_step_w)

    num_extra_pixels_h = tile_step_h * num_tiles_h + overlap_h - image_h
    num_extra_pixels_w = tile_step_w * num_tiles_w + overlap_w - image_w

    margin_bottom = math.floor(num_extra_pixels_h / 2)
    margin_top = math.ceil(num_extra_pixels_h / 2)
    margin_left = math.ceil(num_extra_pixels_w / 2)
    margin_right = math.floor(num_extra_pixels_w / 2)

    margins = Edges(
        bottom=margin_bottom,
        top=margin_top,
        left=margin_left,
        right=margin_right,
    )

    return margins


class TiledImageDataset(Dataset):
    """
    Dataset for storing tiled parts of input image.
    """

    def __init__(
        self,
        *,
        images: List[Union[str, Path]],
        masks: Optional[List[Union[str, Path]]] = None,
        train: bool,
        tile_size: Union[int, Tuple[int]],
        tile_step: Union[int, Tuple[int]],
        input_key: str = "features",
        output_key: str = "targets",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            images: list of paths to image files
            masks: list of paths to mask files
            train: train or infer mode, needs to be provided for appropriate
                choice of image bordering type.
            tile_size: tile size
            tile_step: tile step
            input_key: input key in State for obtaining image tiles
            output_key: output key in State for obtaining masks
            transform: transform for tile
        """
        super().__init__()

        self.images = images
        self.masks = masks
        self.train = train
        self.input_key = input_key
        self.output_key = output_key
        self.transform = transform

        self._init_shapes()
        self._init_tile_size(tile_size)
        self._init_tile_step(tile_step)
        self._check_tile_params()
        self._init_tiles()
        self._init_index()

    def _init_shapes(self):
        self.shapes = []

        if self.masks is None:
            for image_path in self.images:
                image = skimage.io.imread(image_path)
                image_h, image_w, *_ = image.shape

                self.shapes.append(Shape(height=image_h, width=image_w))
        else:
            for image_path, mask_path in zip(self.images, self.masks):
                image = skimage.io.imread(image_path)
                image_h, image_w, *_ = image.shape

                mask = skimage.io.imread(mask_path)
                mask_h, mask_w, *_ = mask.shape

                assert image_h == mask_h and image_w == mask_w, (
                    f"Image and mask shapes differ in "
                    f"image: {image_path} ({image_h}, {image_w}); "
                    f"mask: {mask_path} ({mask_h}, {mask_w})."
                )

                self.shapes.append(Shape(height=image_h, width=image_w))

    def _init_tile_size(self, tile_size: Union[int, Tuple[int]]):
        error_msg = (
            "`tile_size` must be 2-dimensional or scalar, "
            "got {ndim} dimensions."
        )

        if isinstance(tile_size, (tuple, list)):
            tile_size_ndim = len(tile_size)
            assert tile_size_ndim == 2, error_msg.format(
                variable="tile_size", ndim=tile_size_ndim,
            )
            tile_size_h, tile_size_w = tile_size
        else:
            tile_size_h, tile_size_w = tile_size, tile_size

        self.tile_size = Shape(height=tile_size_h, width=tile_size_w)

    def _init_tile_step(self, tile_step: Union[int, Tuple[int]]):
        error_msg = (
            "`tile_size` must be 2-dimensional or scalar, "
            "got {ndim} dimensions."
        )

        if isinstance(tile_step, (tuple, list)):
            tile_step_ndim = len(tile_step)
            assert tile_step_ndim == 2, error_msg.format(
                variable="tile_step", ndim=tile_step_ndim,
            )
            tile_step_h, tile_step_w = tile_step
        else:
            tile_step_h, tile_step_w = tile_step, tile_step

        self.tile_step = Shape(height=tile_step_h, width=tile_step_w)

    def _check_tile_params(self):
        error_msg = (
            "Tile step in {dim} ({tile_step}) "
            "must be less or equal than "
            "tile size in {dim} ({tile_size})."
        )

        assert (
            self.tile_step.height <= self.tile_size.height
        ), error_msg.format(
            dim="height",
            tile_step=self.tile_step.height,
            tile_size=self.tile_size.height,
        )

        assert self.tile_step.width <= self.tile_size.width, error_msg.format(
            dim="width",
            tile_step=self.tile_step.width,
            tile_size=self.tile_size.width,
        )

    def _init_tiles(self):
        self.margins = []
        self.crops = []

        overlap = Shape(
            height=self.tile_size.height - self.tile_step.height,
            width=self.tile_size.width - self.tile_step.width,
        )

        for shape in self.shapes:
            margins = get_image_margins(
                shape.height,
                shape.width,
                self.tile_size.height,
                self.tile_size.width,
                self.tile_step.height,
                self.tile_step.width,
            )
            self.margins.append(margins)
            image_shape_ext = Shape(
                height=shape.height + margins.bottom + margins.top,
                width=shape.width + margins.left + margins.right,
            )

            x_stop = image_shape_ext.width - overlap.width
            y_stop = image_shape_ext.height - overlap.height

            x = torch.arange(0, x_stop, self.tile_step.width)
            y = torch.arange(0, y_stop, self.tile_step.height)

            crops = [
                tuple(map(lambda t: t.item(), pair))
                for pair in torch.cartesian_prod(x, y)
            ]
            self.crops.append(crops)

    def _init_index(self):
        self.num_tiles_cumulative = []
        cumulative = 0
        for crops in self.crops:
            num_tiles = len(crops)
            cumulative += num_tiles
            self.num_tiles_cumulative.append(cumulative)

    def _extend_image_with_margins(self, image: np.ndarray, margins: Edges):
        border_type = cv2.BORDER_DEFAULT if self.train else cv2.BORDER_CONSTANT
        image = cv2.copyMakeBorder(
            image,
            margins.top,
            margins.bottom,
            margins.left,
            margins.right,
            borderType=border_type,
        )

        return image

    def __len__(self):
        """
        Returns length of dataset, i.e. number of tiles.
        """
        return self.num_tiles_cumulative[-1]

    def __getitem__(self, idx: int):
        """
        Args:
            idx: index of element in dataset

        Returns:
            OrderedDict with tile and starting x, y coordinates.
        """
        image_idx = bisect_right(self.num_tiles_cumulative, idx)
        crop_idx = idx - self.num_tiles_cumulative[image_idx]
        x, y = self.crops[image_idx][crop_idx]
        margins = self.margins[image_idx]

        image = skimage.io.imread(self.images[image_idx])
        image = self._extend_image_with_margins(image, margins)
        height, width, *_ = image.shape
        storage_size = Shape(height=height, width=width)
        image = image[
            y : y + self.tile_size.height, x : x + self.tile_size.width
        ]
        dict_ = {"image": image}

        if self.masks is not None:
            mask = skimage.io.imread(self.masks[image_idx])
            mask = self._extend_image_with_margins(mask, margins)
            mask = mask[
                y : y + self.tile_size.height, x : x + self.tile_size.width
            ]
            dict_.update(mask=mask)

        if self.transform is not None:
            if isinstance(self.transform, (BasicTransform, BaseCompose)):
                transform = self.transform(**dict_)
            else:
                transform = self.transform(dict_)
            image = transform["image"]
            if self.masks is not None:
                mask = transform["mask"]

        dict_ = {self.input_key: image}
        if self.masks is not None:
            dict_[self.output_key] = mask

        item = OrderedDict(
            **dict_,
            id=image_idx,
            x=x,
            y=y,
            storage_size=np.array(storage_size),
            margins=np.array(margins),
            tile_size=np.array(self.tile_size),
        )

        return item
