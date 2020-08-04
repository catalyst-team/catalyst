from typing import Union
from pathlib import Path

import torch
from torch.nn import Module


def save_quantized_model(
    model: Module,
    logdir: Union[str, Path] = None,
    checkpoint_name: str = None,
    out_dir: Union[str, Path] = None,
    out_model: Union[str, Path] = None,
) -> None:
    """Saves quantized model.

    Args:
        model (ScriptModule): Traced model
        logdir (Union[str, Path]): Path to experiment
        out_dir (Union[str, Path]): Directory to save model to
            (overrides logdir)
        out_model (Union[str, Path]): Path to save model to
            (overrides logdir & out_dir)

    Raises:
        ValueError: if nothing out of `logdir`, `out_dir` or `out_model`
          is specified.
    """
    if out_model is None:
        file_name = f"{checkpoint_name}_quantized.pth"

        output: Path = out_dir
        if output is None:
            if logdir is None:
                raise ValueError(
                    "One of `logdir`, `out_dir` or `out_model` "
                    "should be specified"
                )
            output: Path = Path(logdir) / "quantized"

        output.mkdir(exist_ok=True, parents=True)

        out_model = str(output / file_name)
    else:
        out_model = str(out_model)

    torch.save(model.state_dict(), out_model)
