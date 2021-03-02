# def format_metric(name: str, value: float) -> str:
#     """Format metric.
#
#     Metric will be returned in the scientific format if 4
#     decimal chars are not enough (metric value lower than 1e-4).
#
#     Args:
#         name: metric name
#         value: value of metric
#
#     Returns:
#         str: formatted metric
#     """
#     if value < 1e-4:
#         return f"{name}={value:1.3e}"
#     return f"{name}={value:.4f}"
import mlflow
from catalyst.typing import Number, Directory, File, Image
from typing import Dict, Any, Optional

EXPERIMENT_PARAMS = {'shared', 'args', 'runner', 'engine', 'model', 'stages',}
STAGE_PARAMS = {'data', 'criterion', 'optimizer', 'scheduler', 'stage'}
EXCLUDE_PARAMS = {'loggers', 'transform', 'callbacks'}


def mlflow_get_or_start_run(run_name):
    """MLflow function. Gets the active run and gives it a name.
    If active run does not exist, starts a new one.
    """
    active_run = mlflow.active_run()
    if active_run:
        mlflow.set_tag('mlflow.runName', run_name)
        return active_run
    return mlflow.start_run(run_name=run_name)


def mlflow_log_dict(
    dictionary: Dict[str, Any], prefix: str = '', log_type: Optional[str] = None
):
    """MLflow function. Logs any value by its type from dictionary recursively.

    Args:
        dictionary: Values to log as dictionary.
        prefix: Prefix for parameter name (if the parameter is composite).
        log_type: The entity of logging (param, metric, artifact, image, etc.).
    """
    for name, value in dictionary.items():
        if name in EXCLUDE_PARAMS:
            continue

        name = name.replace('*', '')
        if prefix not in STAGE_PARAMS and prefix:
            name = f'{prefix}/{name}'

        if log_type == 'dict':
            mlflow.log_dict(dictionary, name)
        elif isinstance(value, dict):
            mlflow_log_dict(value, name, log_type)
        elif log_type == 'param':
            try:
                mlflow.log_param(name, value)
            except mlflow.exceptions.MlflowException:
                continue
        elif isinstance(value, (Directory, File)) or log_type == 'artifact':
            mlflow.log_artifact(value)
        elif isinstance(value, Number):
            mlflow.log_metric(name, value)
        elif isinstance(value, Image):
            mlflow.log_image(value, f'{name}.png')
        else:
            raise ValueError(f'Unknown type of logging value: {type(value)}')


__all__ = [
    "EXPERIMENT_PARAMS",
    "STAGE_PARAMS",
    "EXCLUDE_PARAMS",
    "mlflow_log_dict",
    "mlflow_get_or_start_run"
]
