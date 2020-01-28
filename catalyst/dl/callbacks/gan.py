from typing import Any, Callable, Dict, List, Optional, Union  # isort:skip

from catalyst.core import CriterionCallback, OptimizerCallback
from catalyst.dl import Callback, CallbackOrder, State


"""
MetricCallbacks alternatives for input/output keys
"""


class MultiKeyMetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """

    # TODO:
    #  merge it with MetricCallback in catalyst.core
    #  maybe after the changes with CriterionCallback will be finalized
    #  in the main repo
    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: Optional[Union[str, List[str]]] = "targets",
        output_key: Optional[Union[str, List[str]]] = "logits",
        **metric_params
    ):
        """

        :param prefix:
        :param metric_fn:
        :param input_key:
        :param output_key:
        :param metric_params:
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    @staticmethod
    def _get(dictionary: dict, keys: Optional[Union[str, List[str]]]) -> Any:
        if keys is None:
            result = dictionary
        elif isinstance(keys, list):
            result = {key: dictionary[key] for key in keys}
        else:
            result = dictionary[keys]
        return result

    def on_batch_end(self, state: State):
        """On batch end call"""
        outputs = self._get(state.output, self.output_key)
        targets = self._get(state.input, self.input_key)
        metric = self.metric_fn(outputs, targets, **self.metric_params)
        state.metrics.add_batch_value(name=self.prefix, value=metric)


class WassersteinDistanceCallback(MultiKeyMetricCallback):
    """
    Callback to compute Wasserstein distance metric
    """
    def __init__(
        self,
        prefix: str = "wasserstein_distance",
        real_validity_output_key: str = "real_validity",
        fake_validity_output_key: str = "fake_validity"
    ):
        """

        :param prefix:
        :param real_validity_output_key:
        :param fake_validity_output_key:
        """
        super().__init__(
            prefix,
            metric_fn=self.get_wasserstein_distance,
            input_key=None,
            output_key=[real_validity_output_key, fake_validity_output_key]
        )
        self.real_validity_key = real_validity_output_key
        self.fake_validity_key = fake_validity_output_key

    def get_wasserstein_distance(self, outputs, targets):
        """
        Computes Wasserstein distance
        :param outputs:
        :param targets:
        :return:
        """
        real_validity = outputs[self.real_validity_key]
        fake_validity = outputs[self.fake_validity_key]
        return real_validity.mean() - fake_validity.mean()


"""
CriterionCallback extended
"""


class GradientPenaltyCallback(CriterionCallback):
    """
    Criterion Callback to compute Gradient Penalty
    """

    def __init__(
        self,
        real_input_key: str = "data",
        fake_output_key: str = "fake_data",
        condition_keys: List[str] = None,
        critic_model_key: str = "critic",
        critic_criterion_key: str = "critic",
        real_data_criterion_key: str = "real_data",
        fake_data_criterion_key: str = "fake_data",
        condition_args_criterion_key: str = "critic_condition_args",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0,
    ):
        """

        :param real_input_key: real data key in state.input
        :param fake_output_key: fake data key in state.output
        :param condition_keys: all condition keys in state.input for critic
        :param critic_model_key: key for critic model in state.model
        :param critic_criterion_key: key for critic model in criterion
        :param real_data_criterion_key: key for real data in criterion
        :param fake_data_criterion_key: key for fake data in criterion
        :param condition_args_criterion_key: key for all condition args
            in criterion
        :param prefix:
        :param criterion_key:
        :param multiplier:
        """
        super().__init__(
            input_key=real_input_key,
            output_key=fake_output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            multiplier=multiplier
        )
        self.condition_keys = condition_keys or []
        self.critic_model_key = critic_model_key
        self.critic_criterion_key = critic_criterion_key
        self.real_data_criterion_key = real_data_criterion_key
        self.fake_data_criterion_key = fake_data_criterion_key
        self.condition_args_criterion_key = condition_args_criterion_key

    def _compute_loss(self, state: State, criterion):
        criterion_kwargs = {
            self.real_data_criterion_key: state.input[self.input_key],
            self.fake_data_criterion_key: state.output[self.output_key],
            self.critic_criterion_key: state.model[self.critic_model_key],
            self.condition_args_criterion_key: [
                state.input[key] for key in self.condition_keys
            ]
        }
        return criterion(**criterion_kwargs)


"""
Optimizer Callback with weights clamp after update
"""


class WeightClampingOptimizerCallback(OptimizerCallback):
    """
    Optimizer callback + weights clipping after step is finished
    """
    def __init__(
        self,
        grad_clip_params: Dict = None,
        accumulation_steps: int = 1,
        optimizer_key: str = None,
        loss_key: str = "loss",
        decouple_weight_decay: bool = True,
        weight_clamp_value: float = 0.01
    ):
        """

        :param grad_clip_params:
        :param accumulation_steps:
        :param optimizer_key:
        :param loss_key:
        :param decouple_weight_decay:
        :param weight_clamp_value:
            value to clamp weights after each optimization iteration
            Attention: will clamp WEIGHTS, not GRADIENTS
        """
        super().__init__(
            grad_clip_params=grad_clip_params,
            accumulation_steps=accumulation_steps,
            optimizer_key=optimizer_key,
            loss_key=loss_key,
            decouple_weight_decay=decouple_weight_decay
        )
        self.weight_clamp_value = weight_clamp_value

    def on_batch_end(self, state: State):
        """On batch end event"""
        super().on_batch_end(state)
        if not state.need_backward:
            return

        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )

        need_gradient_step = \
            self._accumulation_counter % self.accumulation_steps == 0

        if need_gradient_step:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.data.clamp_(
                        min=-self.weight_clamp_value,
                        max=self.weight_clamp_value
                    )


__all__ = [
    "WassersteinDistanceCallback",
    "GradientPenaltyCallback",
    "WeightClampingOptimizerCallback"
]
