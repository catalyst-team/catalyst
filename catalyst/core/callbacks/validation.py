from collections import defaultdict

from catalyst.core import _State, Callback, CallbackNode, CallbackOrder


class ValidationManagerCallback(Callback):
    def __init__(self):
        super().__init__(
            order=CallbackOrder.MetricAggregation + 1,
            node=CallbackNode.All,
        )

    def on_epoch_start(self, state: _State):
        state.valid_metrics = defaultdict(None)
        state.is_best_valid = False

    def on_epoch_end(self, state: _State):
        if state.stage_name.startswith("infer") or state.is_distributed_worker:
            return

        state.valid_metrics = {
            k.replace(f"{state.valid_loader}_", ""): v
            for k, v in state.epoch_metrics.items()
            if k.startswith(state.valid_loader)
        }
        assert state.main_metric in state.valid_metrics, \
            f"{state.main_metric} value is not available by the epoch end"

        current_valid_metric = state.valid_metrics[state.main_metric]
        if state.minimize_metric:
            best_valid_metric = \
                state.best_valid_metrics.get(state.main_metric, float("+inf"))
            is_best = current_valid_metric < best_valid_metric
        else:
            best_valid_metric = \
                state.best_valid_metrics.get(state.main_metric, float("-inf"))
            is_best = current_valid_metric > best_valid_metric

        if is_best:
            state.is_best_valid = True
            state.best_valid_metrics = state.valid_metrics.copy()


__all__ = ["ValidationManagerCallback"]
