"""
code modified from https://github.com/belskikh/kekas/blob/master/kekas/utils.py
"""

from typing import List, Dict, Union, Optional
from collections import defaultdict
from pathlib import Path
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator, ScalarEvent
import logging

logging.getLogger("tensorboard").addFilter(lambda x: 0)
logging.getLogger("tensorflow").addFilter(lambda x: 0)


def get_tensorboard_scalars(
    logdir: Union[str, Path], metrics: Optional[List[str]], step: str
) -> Dict[str, List]:
    event_acc = EventAccumulator(str(logdir))
    event_acc.Reload()

    if metrics is not None:
        scalar_names = [
            n for n in event_acc.Tags()["scalars"]
            if step in n and any(m in n for m in metrics)
        ]
    else:
        scalar_names = [n for n in event_acc.Tags()["scalars"] if step in n]

    scalars = {sn: event_acc.Scalars(sn) for sn in scalar_names}
    return scalars


def get_scatter(scalars: List[ScalarEvent], name: str) -> go.Scatter:
    xs = [s.step for s in scalars]
    ys = [s.value for s in scalars]

    return go.Scatter(x=xs, y=ys, name=name)


def plot_tensorboard_log(
    logdir: Union[str, Path],
    step: Optional[str] = "batch",
    metrics: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> None:
    init_notebook_mode(connected=True)
    logdir = Path(logdir)

    logdirs = {
        x.name.replace("_log", ""): x
        for x in logdir.glob("**/*") if x.is_dir() and str(x).endswith("_log")
    }

    scalars_per_loader = {
        key: get_tensorboard_scalars(value, metrics, step)
        for key, value in logdirs.items()
    }

    scalars_per_metric = defaultdict(lambda: {})
    for key, value in scalars_per_loader.items():
        for key2, value2 in value.items():
            scalars_per_metric[key2][key] = value2

    for metric_name, metric_logs in scalars_per_metric.items():
        metric_data = []
        for key, value in metric_logs.items():
            try:
                data_ = get_scatter(value, f"{key}/{metric_name}")
                metric_data.append(data_)
            except:  # noqa: E722
                pass

        layout = go.Layout(
            title=metric_name,
            height=height,
            width=width,
            yaxis=dict(hoverformat=".5f")
        )
        iplot(go.Figure(data=metric_data, layout=layout))
