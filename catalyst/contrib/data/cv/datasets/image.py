from typing import Callable, Optional, Tuple, Union
from collections import OrderedDict
import math
from pathlib import Path

import cv2
import skimage.io

import torch
from torch.utils.data import Dataset


def get_image_margins(
    image_h: int,
    image_w: int,
    tile_size_h: int,
    tile_size_w: int,
    tile_step_h: int,
    tile_step_w: int,
):
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
    overlap_h = tile_size_h - tile_step_h
    overlap_w = tile_size_w - tile_step_w

    num_tiles_h = math.ceil((image_h - overlap_h) / tile_step_h)
    num_tiles_w = math.ceil((image_w - overlap_w) / tile_step_w)

    num_extra_pixels_h = tile_step_h * num_tiles_h + overlap_h - image_h
    num_extra_pixels_w = tile_step_w * num_tiles_w + overlap_w - image_w

    margin_bottom = math.floor(num_extra_pixels_h / 2)
    margin_top = math.ceil(num_extra_pixels_h / 2)
    margin_left = math.floor(num_extra_pixels_w / 2)
    margin_right = math.floor(num_extra_pixels_w / 2)

    margins = OrderedDict(
        margin_bottom=margin_bottom,
        margin_top=margin_top,
        margin_left=margin_left,
        margin_right=margin_right,
    )

    return margins


class TiledImageDataset(Dataset):
    """
    Dataset for storing tiled parts of input image.
    """

    def __init__(
        self,
        image_path: Union[str, Path],
        tile_size: Union[int, Tuple[int]],
        tile_step: Union[int, Tuple[int]],
        input_key: str = "features",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_path: input image path
            tile_size: tile size
            tile_step: tile step
            input_key: input key in State for obtaining image tiles
            transform: optional callable of transforms
        """
        super().__init__()
        image = skimage.io.imread(image_path)
        self.image_h, self.image_w, _ = image.shape
        self.input_key = input_key
        self.transform = transform

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

        error_msg = (
            "Tile step in {dim} ({tile_step}) "
            "must be less or equal than "
            "tile size in {dim} ({tile_size})."
        )

        if self.tile_step_h > self.tile_size_h:
            raise ValueError(
                error_msg.format(
                    dim="height",
                    tile_step=self.tile_step_h,
                    tile_size=self.tile_size_h,
                )
            )

        if self.tile_step_w > self.tile_size_w:
            raise ValueError(
                error_msg.format(
                    dim="width",
                    tile_step=self.tile_step_w,
                    tile_size=self.tile_size_w,
                )
            )

        overlap_h = self.tile_size_h - self.tile_step_h
        overlap_w = self.tile_size_w - self.tile_step_w

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

        image = cv2.copyMakeBorder(
            image,
            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        self.image = image

        x_stop = (
            self.image_w + self.margin_left + self.margin_right - overlap_w
        )
        y_stop = (
            self.image_h + self.margin_bottom + self.margin_top - overlap_h
        )

        x = torch.arange(0, x_stop, self.tile_step_w)
        y = torch.arange(0, y_stop, self.tile_step_h)

        self.crops = torch.cartesian_prod(x, y)

    def __len__(self):
        """
        Returns length of dataset, i.e. number of tiles.
        """
        return self.crops.size(0)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: index of element in dataset

        Returns:
            OrderedDict with tile and starting x, y coordinates.
        """
        x, y = map(lambda coord: coord.item(), self.crops[idx])

        tile = self.image[y : y + self.tile_size_h, x : x + self.tile_size_w]

        if self.transform is not None:
            tile = self.transform({"image": tile})["image"]

        item = OrderedDict(**{self.input_key: tile}, x=x, y=y)

        return item
