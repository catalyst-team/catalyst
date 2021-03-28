# from typing import List
#
# import numpy as np
# import torch
#
# from catalyst.callbacks.criterion import CriterionCallback
# from catalyst.core.runner import IRunner
#
#
# class MixupCallback(CriterionCallback):
#     """Callback to do mixup augmentation.
#
#     More details about mixin can be found in the paper
#     `mixup: Beyond Empirical Risk Minimization`_.
#
#     .. warning::
#         `catalyst.contrib.callbacks.MixupCallback` is inherited from
#         `catalyst.callbacks.CriterionCallback` and does its work.
#         You may not use them together.
#
#     .. _mixup\: Beyond Empirical Risk Minimization:  # noqa: W605
#         https://arxiv.org/abs/1710.09412
#     """
#
#     def __init__(
#         self,
#         input_key: str = "targets",
#         output_key: str = "logits",
#         fields: List[str] = ("features"),
#         alpha=1.0,
#         on_train_only=True,
#         **kwargs
#     ):
#         """
#         Args:
#             fields: list of features which must be affected.
#             alpha: beta distribution a=b parameters.
#                 Must be >=0. The more alpha closer to zero
#                 the less effect of the mixup.
#             on_train_only: Apply to train only.
#                 As the mixup use the proxy inputs, the targets are also proxy.
#                 We are not interested in them, are we?
#                 So, if on_train_only is True, use a standard output/metric
#                 for validation.
#         """
#         assert isinstance(input_key, str) and isinstance(output_key, str)
#         assert len(fields) > 0, "At least one field for MixupCallback is required"
#         assert alpha >= 0, "alpha must be>=0"
#
#         super().__init__(input_key=input_key, input_key=output_key, **kwargs)
#
#         self.on_train_only = on_train_only
#         self.fields = fields
#         self.alpha = alpha
#         self.lam = 1
#         self.index = None
#         self.is_needed = True
#
#     def _compute_loss_value(self, runner: "IRunner", criterion):
#         if not self.is_needed:
#             return super()._compute_loss_value(runner, criterion)
#
#         pred = runner.output[self.input_key]
#         y_a = runner.input[self.input_key]
#         y_b = runner.input[self.input_key][self.index]
#
#         loss = self.lam * criterion(pred, y_a) + (1 - self.lam) * criterion(pred, y_b)
#         return loss
#
#     def on_loader_start(self, runner: "IRunner"):
#         """Loader start hook.
#
#         Args:
#             runner: current runner
#         """
#         self.is_needed = not self.on_train_only or runner.is_train_loader
#
#     def on_batch_start(self, runner: "IRunner") -> None:
#         """Batch start hook.
#
#         Args:
#             runner: current runner
#         """
#         if not self.is_needed:
#             return
#
#         if self.alpha > 0:
#             self.lam = np.random.beta(self.alpha, self.alpha)
#         else:
#             self.lam = 1
#
#         self.index = torch.randperm(runner.input[self.fields[0]].shape[0])
#         self.index.to(runner.device)
#
#         for f in self.fields:
#             runner.input[f] = (
#                 self.lam * runner.input[f] + (1 - self.lam) * runner.input[f][self.index]
#             )
#
#
# __all__ = ["MixupCallback"]
