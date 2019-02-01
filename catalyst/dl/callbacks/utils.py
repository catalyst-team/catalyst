import torch


def get_val_from_metric(metric_value):
    if isinstance(metric_value, (int, float)):
        pass
    elif torch.is_tensor(metric_value):
        metric_value = metric_value.item()
    else:
        metric_value = metric_value.value()
        if isinstance(metric_value, (tuple, list)):
            metric_value = metric_value[0]
        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()
    return metric_value


def process_epoch_metrics(
    epoch_metrics,
    best_metrics,
    valid_loader="valid",
    main_metric="loss",
    minimize=True
):
    valid_metrics = epoch_metrics[valid_loader]
    is_best = True \
        if best_metrics is None \
        else (minimize != (
            valid_metrics[main_metric] > best_metrics[main_metric]))
    best_metrics = valid_metrics if is_best else best_metrics
    return best_metrics, valid_metrics, is_best


def to_batch_metrics(*, state, metric_key, state_key=None):
    metric = state.get_key(state_key or metric_key)
    if isinstance(metric, dict):
        for key, value in metric.items():
            state.batch_metrics[f"{metric_key}_{key}"] = \
                get_val_from_metric(value)
    else:
        state.batch_metrics[f"{metric_key}"] = \
            get_val_from_metric(metric)


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
