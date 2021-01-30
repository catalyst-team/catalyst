from typing import Any, Callable, Dict, List, TYPE_CHECKING, Union
import inspect
import logging
from pathlib import Path

from torch import jit, nn

from catalyst.typing import Device, Model
from catalyst.utils.checkpoint import load_checkpoint, pack_checkpoint, unpack_checkpoint
from catalyst.utils.config import load_config
from catalyst.utils.distributed import assert_fp16_available, get_nn_from_ddp_module
from catalyst.utils.loaders import get_native_batch_from_loaders
from catalyst.utils.misc import get_fn_argsnames
from catalyst.utils.scripts import prepare_config_api_components
from catalyst.utils.torch import any2device, get_requires_grad, set_requires_grad

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner
    from catalyst.experiments import ConfigExperiment

logger = logging.getLogger(__name__)


def _get_input_argnames(fn: Callable[..., Any], exclude: List[str] = None) -> List[str]:
    """
    Function to get input argument names of function.

    Args:
        fn (Callable[..., Any]): Function to get argument names from
        exclude: List of string of names to exclude

    Returns:
        List[str]: List of input argument names
    """
    argspec = inspect.getfullargspec(fn)
    assert argspec.varargs is None and argspec.varkw is None, "not supported by PyTorch"

    return get_fn_argsnames(fn, exclude=exclude)


class _ForwardOverrideModel(nn.Module):
    """Model that calls specified method instead of forward.

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.model, self.method_name)(*args, **kwargs)


class _TracingModelWrapper(nn.Module):
    """Wrapper that traces model with batch instead of calling it.

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name
        self.tracing_result: jit.ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)

        try:
            assert len(args) == 0, "only KV support implemented"

            fn = getattr(self.model, self.method_name)
            method_argnames = _get_input_argnames(fn=fn, exclude=["self"])
            method_input = tuple(kwargs[name] for name in method_argnames)

            self.tracing_result = jit.trace(method_model, method_input)
        except Exception:
            # for backward compatibility
            self.tracing_result = jit.trace(method_model, *args, **kwargs)
        output = self.model.forward(*args, **kwargs)

        return output


def trace_model(
    model: Model,
    predict_fn: Callable,
    batch=None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
    predict_params: dict = None,
) -> jit.ScriptModule:
    """Traces model using runner and batch.

    Args:
        model: Model to trace
        predict_fn: Function to run prediction with the model provided,
            takes model, inputs parameters
        batch: Batch to trace the model
        method_name: Model's method name that will be
            used as entrypoint during tracing
        mode: Mode for model to trace (``train`` or ``eval``)
        requires_grad: Flag to use grads
        opt_level: Apex FP16 init level, optional
        device: Torch device
        predict_params: additional parameters for model forward

    Returns:
        jit.ScriptModule: Traced model

    Raises:
        ValueError: if both batch and predict_fn must be specified or
          mode is not in 'eval' or 'train'.
    """
    if batch is None or predict_fn is None:
        raise ValueError("Both batch and predict_fn must be specified.")

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    predict_params = predict_params or {}

    tracer = _TracingModelWrapper(model, method_name)
    if opt_level is not None:
        assert_fp16_available()
        # If traced in AMP we need to initialize the model before calling
        # the jit
        # https://github.com/NVIDIA/apex/issues/303#issuecomment-493142950
        from apex import amp

        model = model.to(device)
        model = amp.initialize(model, optimizers=None, opt_level=opt_level)

    getattr(model, mode)()
    set_requires_grad(model, requires_grad=requires_grad)

    predict_fn(tracer, batch, **predict_params)

    return tracer.tracing_result


def trace_model_from_checkpoint(
    logdir: Path,
    method_name: str,
    checkpoint_name: str,
    stage: str = None,
    loader: Union[str, int] = None,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
):
    """
    Traces model using created experiment and runner.

    Args:
        logdir (Union[str, Path]): Path to Catalyst logdir with model
        checkpoint_name: Name of model checkpoint to use
        stage: experiment's stage name
        loader (Union[str, int]): experiment's loader name or its index
        method_name: Model's method name that will be
            used as entrypoint during tracing
        mode: Mode for model to trace (``train`` or ``eval``)
        requires_grad: Flag to use grads
        opt_level: AMP FP16 init level
        device: Torch device

    Returns:
        the traced model
    """
    config_path = logdir / "configs" / "_config.json"
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
    logging.info("Load config")
    config: Dict[str, dict] = load_config(config_path)

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    logger.info("Import experiment and runner from logdir")
    experiment: ConfigExperiment = None
    experiment, runner, _ = prepare_config_api_components(expdir=expdir, config=config)

    logger.info(f"Load model state from checkpoints/{checkpoint_name}.pth")
    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = load_checkpoint(checkpoint_path)
    unpack_checkpoint(checkpoint, model=model)
    runner.model, runner.device = model, device

    if loader is None:
        loader = 0
    batch = get_native_batch_from_loaders(loaders=experiment.get_loaders(stage), loader=loader)

    # function to run prediction on batch
    def predict_fn(model, inputs, **kwargs):  # noqa: WPS442
        model_dump = runner.model
        runner.model = model
        result = runner.predict_batch(inputs, **kwargs)
        runner.model = model_dump
        return result

    logger.info("Tracing is running...")
    traced_model = trace_model(
        model=model,
        predict_fn=predict_fn,
        batch=batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    logger.info("Done")
    return traced_model


def trace_model_from_runner(
    runner: "IRunner",
    checkpoint_name: str = None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
) -> jit.ScriptModule:
    """
    Traces model using created experiment and runner.

    Args:
        runner: current runner.
        checkpoint_name: Name of model checkpoint to use, if None
            traces current model from runner
        method_name: Model's method name that will be
            used as entrypoint during tracing
        mode: Mode for model to trace (``train`` or ``eval``)
        requires_grad: Flag to use grads
        opt_level: AMP FP16 init level
        device: Torch device

    Returns:
        ScriptModule: Traced model
    """
    logdir = runner.logdir
    model = get_nn_from_ddp_module(runner.model)

    if checkpoint_name is not None:
        dumped_checkpoint = pack_checkpoint(model=model)
        checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
        checkpoint = load_checkpoint(filepath=checkpoint_path)
        unpack_checkpoint(checkpoint=checkpoint, model=model)

    # getting input names of args for method since we don't have Runner
    # and we don't know input_key to preprocess batch for method call
    fn = getattr(model, method_name)
    method_argnames = _get_input_argnames(fn=fn, exclude=["self"])

    batch = {}
    for name in method_argnames:
        # TODO: We don't know input_keys without runner
        assert name in runner.input, (
            "Input batch should contain the same keys as input argument "
            "names of `forward` function to be traced correctly"
        )
        batch[name] = runner.input[name]

    batch = any2device(batch, device)

    # Dumping previous runner of the model, we will need it to restore
    device_dump, is_training_dump, requires_grad_dump = (
        runner.device,
        model.training,
        get_requires_grad(model),
    )

    model.to(device)

    # Function to run prediction on batch
    def predict_fn(model: Model, inputs, **kwargs):  # noqa: WPS442
        return model(**inputs, **kwargs)

    traced_model = trace_model(
        model=model,
        predict_fn=predict_fn,
        batch=batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    if checkpoint_name is not None:
        unpack_checkpoint(checkpoint=dumped_checkpoint, model=model)

    # Restore previous runner of the model
    getattr(model, "train" if is_training_dump else "eval")()
    set_requires_grad(model, requires_grad_dump)
    model.to(device_dump)

    return traced_model


def get_trace_name(
    method_name: str,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    additional_string: str = None,
) -> str:
    """Creates a file name for the traced model.

    Args:
        method_name: model's method name
        mode: ``train`` or ``eval``
        requires_grad: flag if model was traced with gradients
        opt_level: opt_level if model was traced in FP16
        additional_string: any additional information

    Returns:
        str: Filename for traced model to be saved.
    """
    file_name = "traced"
    if additional_string is not None:
        file_name += f"-{additional_string}"

    file_name += f"-{method_name}"
    if mode == "train":
        file_name += "-in_train"

    if requires_grad:
        file_name += "-with_grad"

    if opt_level is not None:
        file_name += "-opt_{opt_level}"

    file_name += ".pth"

    return file_name


def save_traced_model(
    model: jit.ScriptModule,
    logdir: Union[str, Path] = None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    out_dir: Union[str, Path] = None,
    out_model: Union[str, Path] = None,
    checkpoint_name: str = None,
) -> None:
    """Saves traced model.

    Args:
        model: Traced model
        logdir (Union[str, Path]): Path to experiment
        method_name: Name of the method was traced
        mode: Model's mode - `train` or `eval`
        requires_grad: Whether model was traced with require_grad or not
        opt_level: Apex FP16 init level used during tracing
        out_dir (Union[str, Path]): Directory to save model to
            (overrides logdir)
        out_model (Union[str, Path]): Path to save model to
            (overrides logdir & out_dir)
        checkpoint_name: Checkpoint name used to restore the model

    Raises:
        ValueError: if nothing out of `logdir`, `out_dir` or `out_model`
          is specified.
    """
    if out_model is None:
        file_name = get_trace_name(
            method_name=method_name,
            mode=mode,
            requires_grad=requires_grad,
            opt_level=opt_level,
            additional_string=checkpoint_name,
        )

        output: Path = out_dir
        if output is None:
            if logdir is None:
                raise ValueError(
                    "One of `logdir`, `out_dir` or `out_model` " "should be specified"
                )
            output: Path = Path(logdir) / "trace"

        output.mkdir(exist_ok=True, parents=True)

        out_model = str(output / file_name)
    else:
        out_model = str(out_model)

    jit.save(model, out_model)


def load_traced_model(
    model_path: Union[str, Path], device: Device = "cpu", opt_level: str = None,
) -> jit.ScriptModule:
    """Loads a traced model.

    Args:
        model_path: Path to traced model
        device: Torch device
        opt_level: Apex FP16 init level, optional

    Returns:
        ScriptModule: Traced model
    """
    # jit.load dont work with pathlib.Path
    model_path = str(model_path)

    if opt_level is not None:
        device = "cuda"

    model = jit.load(model_path, map_location=device)

    if opt_level is not None:
        assert_fp16_available()
        from apex import amp

        model = amp.initialize(model, optimizers=None, opt_level=opt_level)

    return model


__all__ = [
    "trace_model",
    "trace_model_from_checkpoint",
    "trace_model_from_runner",
    "get_trace_name",
    "save_traced_model",
    "load_traced_model",
]
