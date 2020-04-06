from typing import Dict, List, Optional

from catalyst.core import CriterionCallback, OptimizerCallback
from catalyst.dl import MetricCallback, State

# MetricCallbacks alternatives for input/output keys


class WassersteinDistanceCallback(MetricCallback):
    """Callback to compute Wasserstein distance metric."""

    def __init__(
        self,
        prefix: str = "wasserstein_distance",
        real_validity_output_key: str = "real_validity",
        fake_validity_output_key: str = "fake_validity",
        multiplier: float = 1.0,
    ):
        """
        Args:
            prefix (str):
            real_validity_output_key (str):
            fake_validity_output_key (str):
        """
        super().__init__(
            prefix,
            metric_fn=self.get_wasserstein_distance,
            input_key={},
            output_key={
                real_validity_output_key: "real_validity",
                fake_validity_output_key: "fake_validity",
            },
            multiplier=multiplier,
        )

    def get_wasserstein_distance(self, real_validity, fake_validity):
        """Computes Wasserstein distance."""
        return real_validity.mean() - fake_validity.mean()


# CriterionCallback extended


class GradientPenaltyCallback(CriterionCallback):
    """Criterion Callback to compute Gradient Penalty."""

    def __init__(
        self,
        real_input_key: str = "data",
        fake_output_key: str = "fake_data",
        condition_keys: Optional[List[str]] = None,
        critic_model_key: str = "critic",
        critic_criterion_key: str = "critic",
        real_data_criterion_key: str = "real_data",
        fake_data_criterion_key: str = "fake_data",
        condition_args_criterion_key: str = "critic_condition_args",
        prefix: str = "loss",
        criterion_key: Optional[str] = None,
        multiplier: float = 1.0,
    ):
        """
        Args:
            real_input_key (str): real data key in ``state.input``
            fake_output_key (str): fake data key in ``state.output``
            condition_keys (List[str], optional): all condition keys
                in ``state.input`` for critic
            critic_model_key (str): key for critic model in ``state.model``
            critic_criterion_key (str): key for critic model in criterion
            real_data_criterion_key (str): key for real data in criterion
            fake_data_criterion_key (str): key for fake data in criterion
            condition_args_criterion_key (str): key for all condition args
                in criterion
            prefix (str):
            criterion_key (str):
            multiplier (float):
        """
        super().__init__(
            input_key=real_input_key,
            output_key=fake_output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            multiplier=multiplier,
        )
        self.condition_keys = condition_keys or []
        self.critic_model_key = critic_model_key
        self.critic_criterion_key = critic_criterion_key
        self.real_data_criterion_key = real_data_criterion_key
        self.fake_data_criterion_key = fake_data_criterion_key
        self.condition_args_criterion_key = condition_args_criterion_key

    def _compute_metric(self, state: State):
        criterion_kwargs = {
            self.real_data_criterion_key: state.batch_in[self.input_key],
            self.fake_data_criterion_key: state.batch_out[self.output_key],
            self.critic_criterion_key: state.model[self.critic_model_key],
            self.condition_args_criterion_key: [
                state.batch_in[key] for key in self.condition_keys
            ],
        }
        criterion = state.get_attr("criterion", self.criterion_key)
        return criterion(**criterion_kwargs)


# Optimizer Callback with weights clamp after update


class WeightClampingOptimizerCallback(OptimizerCallback):
    """Optimizer callback + weights clipping after step is finished."""

    def __init__(
        self,
        grad_clip_params: Optional[Dict] = None,
        accumulation_steps: int = 1,
        optimizer_key: Optional[str] = None,
        loss_key: str = "loss",
        decouple_weight_decay: bool = True,
        weight_clamp_value: float = 0.01,
    ):
        """
        Args:
            grad_clip_params (dict, optional):
            accumulation_steps (int):
            optimizer_key (str, optional):
            loss_key (str):
            decouple_weight_decay (bool):
            weight_clamp_value (float): value to clamp weights after each
                optimization iteration

        .. note::
            ``weight_clamp_value`` will clamp WEIGHTS, not GRADIENTS
        """
        super().__init__(
            grad_clip_params=grad_clip_params,
            accumulation_steps=accumulation_steps,
            optimizer_key=optimizer_key,
            loss_key=loss_key,
            decouple_weight_decay=decouple_weight_decay,
        )
        self.weight_clamp_value = weight_clamp_value

    def on_batch_end(self, state: State) -> None:
        """On batch end event.

        Args:
            state (State): current state
        """
        super().on_batch_end(state)
        if not state.is_train_loader:
            return

        optimizer = state.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )

        need_gradient_step = (
            self._accumulation_counter % self.accumulation_steps == 0
        )

        if need_gradient_step:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.data.clamp_(
                        min=-self.weight_clamp_value,
                        max=self.weight_clamp_value,
                    )


__all__ = [
    "WassersteinDistanceCallback",
    "GradientPenaltyCallback",
    "WeightClampingOptimizerCallback",
]
