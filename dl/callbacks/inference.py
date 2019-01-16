import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from .core import Callback
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


class InferCallback(Callback):
    def __init__(self, out_prefix=None):
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])

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
                        suffix=".".join([state.loader_mode, key])
                    ), value
                )


class InferMaskCallback(Callback):
    def __init__(
        self,
        out_prefix=None,
        mean=None,
        std=None,
        mask_type="soft",
        threshold=None,
        input_key=None,
        output_key=None
    ):
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])
        self.mean = mean or np.array([0.485, 0.456, 0.406])
        self.std = std or np.array([0.229, 0.224, 0.225])
        assert mask_type in ["soft", "hard"], mask_type
        self.mask_type = mask_type
        self.threshold = threshold
        assert input_key is not None
        assert output_key is not None
        self.input_key = input_key
        self.output_key = output_key
        self.counter = 0

    def on_loader_start(self, state):
        lm = state.loader_mode
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, state):
        lm = state.loader_mode
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

        for i in range(probs.shape[0]):
            img = np.uint8(255 * (self.std * features[i] + self.mean))
            filename = f"{self.out_prefix}/{lm}/{self.counter}.jpg"
            cv2.imwrite(filename, img)

            for t in range(probs.shape[-1]):
                mask = probs[i, :, :, t] > self.threshold \
                    if self.threshold is not None \
                    else probs[i, :, :, t]
                mask = np.float32(np.expand_dims(mask, -1))

                masked_img = img * mask

                # @TODO: better naming
                filename = f"{self.out_prefix}/{lm}/{self.counter}_{t}.jpg"
                cv2.imwrite(filename, masked_img)
            self.counter += 1
