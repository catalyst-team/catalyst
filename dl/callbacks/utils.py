import torch
from catalyst.utils.factory import UtilsFactory


def to_batch_metrics(*, state, metric_key):
    metric = state.get_key(metric_key)
    if isinstance(metric, dict):
        for key, value in metric.items():
            state.batch_metrics[f"{metric_key}_{key}"] = \
                UtilsFactory.get_val_from_metric(value)
    else:
        state.batch_metrics[f"{metric_key}"] = \
            UtilsFactory.get_val_from_metric(metric)


def get_optimizer_momentum(optimizer):
    if isinstance(optimizer, torch.optim.Adam):
        return list(optimizer.param_groups)[0]["betas"][0]
    elif isinstance(optimizer, torch.optim.SGD):
        return list(optimizer.param_groups)[0]["momentum"]
    else:
        return None


def scheduler_step(scheduler, valid_metric=None):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(valid_metric)
        lr = list(scheduler.optimizer.param_groups)[0]["lr"]
    else:
        scheduler.step()
        lr = scheduler.get_lr()[0]

    momentum = get_optimizer_momentum(scheduler.optimizer)

    return lr, momentum
