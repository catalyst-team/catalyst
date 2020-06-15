import os

import imageio
import numpy as np
from skimage.color import label2rgb

import torch
import torch.nn.functional as F  # noqa: N812, WPS301

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.dl import utils


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
        super().__init__(CallbackOrder.internal)
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
        self._keys_from_runner = ["out_dir", "out_prefix"]

    def on_stage_start(self, runner: IRunner):
        """Stage start hook.

        Args:
            runner (IRunner): current runner
        """
        for key in self._keys_from_runner:
            value = getattr(runner, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        self.out_prefix = (
            self.out_prefix if self.out_prefix is not None else ""
        )
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, runner: IRunner):
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        lm = runner.loader_name
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, runner: IRunner):
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        lm = runner.loader_name
        names = runner.input.get(self.name_key, [])

        features = runner.input[self.input_key].detach().cpu()
        images = utils.tensor_to_ndimage(features)

        logits = runner.output[self.output_key]
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

        for index, (image, mask) in enumerate(zip(images, masks)):
            try:
                suffix = names[index]
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
