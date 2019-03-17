import os
from collections import defaultdict
import random
import numpy as np
from skimage.morphology import erosion, disk
import cv2

import torch
import torch.nn.functional as F

from .core import Callback


# @TODO: refactor
class InferCallback(Callback):
    def __init__(self, out_dir=None, out_prefix=None):
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])
        self._keys_from_state = ["out_dir", "out_prefix"]

    def on_stage_start(self, state):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        if self.out_prefix is not None:
            os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, state):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state):
        dct = state.output
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            self.predictions[key].append(value)

    def on_loader_end(self, state):
        self.predictions = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions.items()
        }
        if self.out_prefix is not None:
            for key, value in self.predictions.items():
                np.save(
                    self.out_prefix.format(
                        suffix=".".join([state.loader_name, key])
                    ), value
                )


class InferMaskCallback(Callback):
    def __init__(
        self,
        out_dir=None,
        out_prefix=None,
        input_key=None,
        output_key=None,
        mean=None,
        std=None,
        mask_type="soft",
        threshold=None,
        name_key=None,
        dump_mask=False,
    ):
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.mean = mean or np.array([0.485, 0.456, 0.406])
        self.std = std or np.array([0.229, 0.224, 0.225])
        assert mask_type in ["soft", "hard"], mask_type
        assert input_key is not None
        assert output_key is not None
        self.mask_type = mask_type
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key
        self.name_key = name_key
        self.dump_mask = dump_mask
        self.counter = 0
        self._keys_from_state = ["out_dir", "out_prefix"]

    def on_stage_start(self, state):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        if self.out_dir is not None:
            self.out_prefix = str(self.out_dir) + "/" + str(self.out_prefix)
        if self.out_prefix is not None:
            os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    @staticmethod
    def _get_spaced_colors2(n_colors, seed=42):
        random.seed(seed)
        r, g, b = [int(random.random() * 256) for _ in range(3)]

        step = 256 / n_colors
        ret = []
        for i in range(n_colors):
            r += step
            g += step
            b += step
            ret.append((int(r) % 256, int(g) % 256, int(b) % 256))
        return ret

    def on_loader_start(self, state):
        lm = state.loader_mode
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, state):
        lm = state.loader_mode
        names = state.input.get(self.name_key, {})

        features = state.input[self.input_key]
        logits = state.output[self.output_key]
        logits = torch.unsqueeze_(logits, dim=1) \
            if len(logits.shape) < 4 \
            else logits

        if self.mask_type == "soft":
            probs = F.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        features = features.detach().cpu().numpy()
        features = np.transpose(features, (0, 2, 3, 1))

        probs = probs.detach().cpu().numpy()
        probs = np.transpose(probs, (0, 2, 3, 1))

        colors = self._get_spaced_colors2(n_colors=probs.shape[3])
        for i in range(probs.shape[0]):
            img = np.uint8(255 * (self.std * features[i] + self.mean))
            suffix = names.get(i, f"{self.counter}:6d")
            self.counter += 1

            shw = img.copy()
            for t in range(probs.shape[3]):
                mask = probs[i, :, :, t] > self.threshold \
                    if self.threshold is not None \
                    else probs[i, :, :, t]
                mask = mask.astype(np.float32)

                if self.dump_mask:
                    mask_ = np.concatenate(
                        [np.expand_dims(mask, -1)] * 3,
                        axis=-1).astype(np.float32) * 255
                    filename_ = f"{self.out_prefix}/{lm}/{suffix}_{t}.jpg"
                    cv2.imwrite(filename_, mask_)

                mask = mask - erosion(mask, disk(4))
                shw[mask > 0.5] = colors[t]

            filename = f"{self.out_prefix}/{lm}/{suffix}.jpg"
            cv2.imwrite(filename, shw[:, :, ::-1])
