import os
from collections import defaultdict
import numpy as np
import imageio
from skimage.color import label2rgb

import torch
import torch.nn.functional as F

from catalyst.dl.core import Callback, RunnerState, CallbackOrder
from catalyst.utils import tensor_to_ndimage


# @TODO: refactor
class InferCallback(Callback):
    def __init__(self, out_dir=None, out_prefix=None):
        super().__init__(CallbackOrder.Internal)
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])
        self._keys_from_state = ["out_dir", "out_prefix"]

    def on_stage_start(self, state: RunnerState):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        if self.out_prefix is not None:
            os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, state: RunnerState):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state: RunnerState):
        dct = state.output
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            self.predictions[key].append(value)

    def on_loader_end(self, state: RunnerState):
        self.predictions = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions.items()
        }
        if self.out_prefix is not None:
            for key, value in self.predictions.items():
                suffix = ".".join([state.loader_name, key])
                np.save(f"{self.out_prefix}/{suffix}.npy", value)


class InferMaskCallback(Callback):
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
        mask_type: str = "soft"
    ):
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

    def on_stage_start(self, state: RunnerState):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        self.out_prefix = self.out_prefix \
            if self.out_prefix is not None \
            else ""
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, state: RunnerState):
        lm = state.loader_name
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, state: RunnerState):
        lm = state.loader_name
        names = state.input.get(self.name_key, [])

        features = state.input[self.input_key].detach().cpu()
        images = tensor_to_ndimage(features)

        logits = state.output[self.output_key]
        logits = torch.unsqueeze_(logits, dim=1) \
            if len(logits.shape) < 4 \
            else logits

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

            image = image * (1 - self.mask_strength) \
                + mask * self.mask_strength
            image = (image * 255).clip(0, 255).round().astype(np.uint8)

            filename = f"{self.out_prefix}/{lm}/{suffix}.jpg"
            imageio.imwrite(filename, image)


__all__ = ["InferCallback", "InferMaskCallback"]
