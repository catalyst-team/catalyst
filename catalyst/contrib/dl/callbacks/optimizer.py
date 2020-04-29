from typing import Dict

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from catalyst.core import Callback, CallbackNode, CallbackOrder, State
from catalyst.tools.typing import Model


class SaveModelGradsCallback(Callback):
    """Callback for logging model gradients."""

    def __init__(
        self, norm_type: int = 2, accumulation_steps: int = 1,
    ):
        """
        Args:
            norm_type (int): norm type used to calculate norm of gradients.
                If `OptimizerCallback` provides non-default argument
                `grad_clip_params` with custom norm type, then corresponding
                norm type should be used in this class.
            accumulation_steps (int): number of steps before
                ``model.zero_grad()``.
                Should be the same as in `OptimizerCallback`.
        """
        super().__init__(
            order=CallbackOrder.Optimizer + 1, node=CallbackNode.All
        )

        self.grad_norm_prefix = "_grad_norm"
        self.norm_type = norm_type

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

    @staticmethod
    def grad_norm(*, model: Model, prefix: str, norm_type: int,) -> Dict:
        """Computes gradient norms for a given model.

        Args:
            model (Model): model which gradients to be saved.
            prefix (str): prefix for keys in resulting dictionary.
            norm_type (int): norm type of gradient norm.
        Returns:
            Dict: dictionary in which gradient norms are stored.
        """
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        total_norm = 0.0
        grad_norm = {}

        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            metrics_tag = f"{prefix}/{tag}"
            param_norm = value.grad.data.norm(norm_type).item()
            total_norm += param_norm ** norm_type
            grad_norm[metrics_tag] = param_norm

        total_norm = total_norm ** (1.0 / norm_type)
        tag = "total"
        metrics_tag = f"{prefix}/{tag}"
        grad_norm[metrics_tag] = total_norm

        return grad_norm

    def on_batch_end(self, state: State) -> None:
        """On batch end event

        Args:
            state (State): current state
        """
        if not state.is_train_loader:
            return

        self._accumulation_counter += 1
        need_gradient_step = (
            self._accumulation_counter % self.accumulation_steps == 0
        )

        if need_gradient_step:
            grad_norm = self.grad_norm(
                model=state.model,
                prefix=self.grad_norm_prefix,
                norm_type=self.norm_type,
            )

            state.batch_metrics.update(**grad_norm)
            self._accumulation_counter = 0
