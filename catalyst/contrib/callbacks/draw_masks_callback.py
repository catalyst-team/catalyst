from typing import List, Optional, TYPE_CHECKING
import os

import numpy as np
from skimage.color import label2rgb

import torch
from torch.utils.tensorboard import SummaryWriter

from catalyst.callbacks import ILoggerCallback
from catalyst.contrib.utils.cv.tensor import tensor_to_ndimage
from catalyst.core.callback import CallbackNode, CallbackOrder

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class DrawMasksCallback(ILoggerCallback):
    """
    Logger callback draw masks for common segmentation task: image -> masks
    """

    def __init__(
        self,
        pred_mask_key: str,
        image_key: Optional[str] = None,
        gt_mask_key: Optional[str] = None,
        mask2show: Optional[List[int]] = None,
        activation: Optional[str] = "Sigmoid",
        log_name: str = "images",
        summary_step: int = 50,
        threshold: float = 0.5,
    ):
        """

        Args:
            pred_mask_key: predicted mask key
            image_key: input image key. If None mask will be drawn on black
            background
            gt_mask_key: ground truth mask key. If None, will not be drawn
            mask2show: mask indexes to show, if None all mask will be drawn. By
            this parameter you can change the mask order
            activation: An torch.nn activation applied to the outputs.
            Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
            log_name: logging name. If you use several such "callbacks", they
            must have different logging names
            summary_step: logging frequency
            threshold: threshold for predicted masks, must be in (0, 1)
        """
        assert 0 < threshold < 1
        assert activation in ["none", "Sigmoid", "Softmax2d"]
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)

        self.image_key = image_key
        self.gt_mask_key = gt_mask_key
        self.pred_mask_key = pred_mask_key

        self.mask2show = mask2show
        self.summary_step = summary_step
        self.threshold = threshold
        self.log_name = log_name

        if activation == "Sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "Softmax2d":
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = lambda x: x

        self.loggers = {}
        self.step = None  # initialization

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook.

        Args:
            runner: current runner
        """
        if runner.loader_key not in self.loggers:
            log_dir = os.path.join(
                runner.logdir, f"{runner.loader_key}_log/images/"
            )
            self.loggers[runner.loader_key] = SummaryWriter(log_dir)
        self.step = 0

    def _draw_masks(
        self,
        writer: SummaryWriter,
        image_over_predicted_mask: np.array,
        image_over_gt_mask: Optional[np.array],
        global_step: int,
    ) -> None:
        """
        Draw image over mask to tensorboard

        Args:
            writer: loader writer
            image_over_predicted_mask: image over predicted mask
            image_over_gt_mask: image over ground truth mask
            global_step: global step
        """
        if image_over_gt_mask is not None:
            writer.add_image(
                f"{self.log_name} Ground Truth",
                image_over_gt_mask,
                global_step=global_step,
                dataformats="HWC",
            )

        writer.add_image(
            f"{self.log_name} Prediction",
            image_over_predicted_mask,
            global_step=global_step,
            dataformats="HWC",
        )

    def _prob2mask(self, prob_masks: np.array) -> np.array:
        """
        Convert probability masks into label mask

        Args:
            prob_masks: [n_classes, H, W], probability masks for each class

        Returns: [H, W] label mask
        """
        mask = np.zeros_like(prob_masks[0], dtype=np.uint8)
        n_classes = mask.shape[0]
        if self.mask2show is not None:
            assert max(self.mask2show) < n_classes
            mask2show = self.mask2show
        else:
            mask2show = range(n_classes)

        for i in mask2show:
            prob_mask = prob_masks[i]
            mask[prob_mask >= self.threshold] = i + 1
        return mask

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook.

        Args:
            runner: current runner
        """
        if self.step % self.summary_step == 0:
            pred_mask = runner.output[self.pred_mask_key][0]
            pred_mask = self.activation(pred_mask)
            pred_mask = pred_mask.cpu().detach().numpy()
            pred_mask = self._prob2mask(pred_mask)

            if self.gt_mask_key is not None:
                gt_mask = runner.input[self.gt_mask_key][0]
                gt_mask = gt_mask.cpu().detach().numpy()
                gt_mask = self._prob2mask(gt_mask)
            else:
                gt_mask = None

            if self.image_key is not None:
                image = runner.input[self.image_key][0].cpu()
                image = tensor_to_ndimage(image)
            else:
                image = np.zeros_like(pred_mask, dtype=np.uint8)

            image_over_predicted_mask = label2rgb(pred_mask, image, bg_label=0)
            if gt_mask is not None:
                image_over_gt_mask = label2rgb(gt_mask, image, bg_label=0)
            else:
                image_over_gt_mask = None

            self._draw_masks(
                self.loggers[runner.loader_key],
                image_over_predicted_mask,
                image_over_gt_mask,
                runner.global_sample_step,
            )
        self.step += 1


__all__ = ["DrawMasksCallback"]
