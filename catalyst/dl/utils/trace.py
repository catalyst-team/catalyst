from typing import TYPE_CHECKING, Any, Dict, List, Callable, Union  # isort:skip
import inspect
from pathlib import Path

from torch import nn
from torch.jit import ScriptModule, trace, load, save

from catalyst.dl import Experiment, State
from catalyst.utils.tools.typing import Device, Model
from catalyst.utils import (
    import_experiment_and_runner,
    assert_fp16_available,
    is_wrapped_with_ddp,
    set_requires_grad,
    get_requires_grad,
    unpack_checkpoint,
    get_native_batch,
    get_fn_argsnames,
    load_checkpoint,
    load_config,
    any2device,
)

if TYPE_CHECKING:
    from catalyst.dl import Runner  # noqa: F401


def get_input_argnames(fn: Callable[..., Any], exclude: List[str] = None):
    argspec = inspect.getfullargspec(fn)
    assert (
            argspec.varargs is None and argspec.varkw is None
    ), "not supported by PyTorch"

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
        self.tracing_result: ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)

        try:
            assert len(args) == 0, "only KV support implemented"

            fn = getattr(self.model, self.method_name)
            method_argnames = get_input_argnames(fn=fn, exclude=["self"])
            method_input = tuple(kwargs[name] for name in method_argnames)

            self.tracing_result = trace(method_model, method_input)
        except Exception:
            # for backward compatibility
            self.tracing_result = trace(
                method_model, *args, **kwargs
            )
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
) -> ScriptModule:
    """Traces model using runner and batch.

    Args:
        model: Model to trace
        predict_fn: Function to run prediction with the model provided,
            takes model, inputs parameters
        batch: Batch to trace the model
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): Apex FP16 init level, optional
        device (str): Torch device
        predict_params (dict): additional parameters for model forward

    Returns:
        (ScriptModule): Traced model
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
        # TODO: remove `check_trace=False`
        # after fixing this bug https://github.com/pytorch/pytorch/issues/23993
        params = {**predict_params, "check_trace": False}
    else:
        params = predict_params

    getattr(model, mode)()
    set_requires_grad(model, requires_grad=requires_grad)

    predict_fn(tracer, batch, **params)

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
        checkpoint_name (str): Name of model checkpoint to use
        stage (str): experiment's stage name
        loader (Union[str, int]): experiment's loader name or its index
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): AMP FP16 init level
        device (str): Torch device

    Returns:
        the traced model
    """
    config_path = logdir / "configs" / "_config.json"
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
    print("Load config")
    config: Dict[str, dict] = load_config(config_path)
    runner_params = config.get("runner_params", {}) or {}

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    print("Import experiment and runner from logdir")
    ExperimentType, RunnerType = import_experiment_and_runner(expdir)
    experiment: Experiment = ExperimentType(config)

    print(f"Load model state from checkpoints/{checkpoint_name}.pth")
    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = load_checkpoint(checkpoint_path)
    unpack_checkpoint(checkpoint, model=model)

    runner: RunnerType = RunnerType(**runner_params)
    runner.model, runner.device = model, device

    if loader is None:
        loader = 0
    batch = get_native_batch(experiment.get_loaders(stage), loader)

    # function to run prediction on batch
    def predict_fn(model, inputs, **kwargs):
        _model = runner.model
        runner.model = model
        result = runner.predict_batch(inputs, **kwargs)
        runner.model = _model
        return result

    print("Tracing")
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

    print("Done")
    return traced_model


def trace_model_from_state(
    state: State,
    method_name: str = "forward",
    loader: Union[str, int] = None,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
) -> ScriptModule:
    """
    Traces model using created experiment and runner.

    Args:
        state (State): Current runner state.
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        loader (Union[str, int]): experiment's loader name or its index
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): AMP FP16 init level
        device (str): Torch device

    Returns:
        (ScriptModule): Traced model
    """
    model = state.model
    if is_wrapped_with_ddp(model):
        model = model.module

    # getting input names of args for method since we don't have Runner
    # and we don't know input_key to preprocess batch for method call
    fn = getattr(model, method_name)
    method_argnames = get_input_argnames(fn=fn, exclude=["self"])

    # getting batch from loaders and filter it by method argnames
    if loader is None:
        loader = 0
    batch = get_native_batch(state.loaders, loader)
    batch = any2device({name: batch[name] for name in method_argnames}, device)

    # Dumping previous state of the model, we will need it to restore
    _device, _is_training, _requires_grad = \
        state.device, model.training, get_requires_grad(model)

    model.to(device)

    # Function to run prediction on batch
    def predict_fn(model: Model, inputs, **kwargs):
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

    # Restore previous state of the model
    getattr(model, "train" if _is_training else "eval")()
    set_requires_grad(model, _requires_grad)
    model.to(_device)

    return traced_model


def get_trace_name(
    method_name: str,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    additional_string: str = None,
):
    """Creates a file name for the traced model.

    Args:
        method_name (str): model's method name
        mode (str): ``train`` or ``eval``
        requires_grad (bool): flag if model was traced with gradients
        opt_level (str): opt_level if model was traced in FP16
        additional_string (str): any additional information
    """
    file_name = f"traced"
    if additional_string is not None:
        file_name += f"-{additional_string}"

    file_name += f"-{method_name}"
    if mode == "train":
        file_name += "-in_train"

    if requires_grad:
        file_name += f"-with_grad"

    if opt_level is not None:
        file_name += f"-opt_{opt_level}"

    file_name += ".pth"

    return file_name


def save_traced_model(
    model: ScriptModule,
    logdir: Union[str, Path] = None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    out_dir: Union[str, Path] = None,
    out_model: Union[str, Path] = None,
    checkpoint_name: str = None,
):
    """Saves traced model.

    Args:
        model (ScriptModule): Traced model
        logdir (Union[str, Path]): Path to experiment
        method_name (str): Name of the method was traced
        mode (str): Model's mode - `train` or `eval`
        requires_grad (bool): Whether model was traced with require_grad or not
        opt_level (str): Apex FP16 init level used during tracing
        out_dir (Union[str, Path]): Directory to save model to
            (overrides logdir)
        out_model (Union[str, Path]): Path to save model to
            (overrides logdir & out_dir)
        checkpoint_name (str): Checkpoint name used to restore the model
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
            output: Path = Path(logdir) / "trace"

        output.mkdir(exist_ok=True, parents=True)

        out_model = str(output / file_name)
    else:
        out_model = str(out_model)

    save(model, out_model)


def load_traced_model(
    model_path: Union[str, Path],
    device: Device = "cpu",
    opt_level: str = None,
) -> ScriptModule:
    """Loads a traced model.

    Args:
        model_path: Path to traced model
        device (str): Torch device
        opt_level (str): Apex FP16 init level, optional

    Returns:
        (ScriptModule): Traced model
    """
    # jit.load dont work with pathlib.Path
    model_path = str(model_path)

    if opt_level is not None:
        device = "cuda"

    model = load(model_path, map_location=device)

    if opt_level is not None:
        assert_fp16_available()
        from apex import amp

        model = amp.initialize(model, optimizers=None, opt_level=opt_level)

    return model


__all__ = [
    "trace_model",
    "trace_model_from_checkpoint",
    "trace_model_from_state",
    "get_trace_name",
    "save_traced_model",
    "load_traced_model",
]
