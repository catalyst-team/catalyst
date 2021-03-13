from typing import Dict, Union
from collections import OrderedDict

import torch
from torch import nn

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine
from catalyst.settings import SETTINGS
from catalyst.typing import RunnerModel, RunnerOptimizer
from catalyst.utils.misc import get_fn_default_params

if SETTINGS.apex_required:
    import apex
    import apex.amp as amp
    from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel


def _initialize_apex(model, optimizer=None, **engine_params):
    """
    Prepares model and optimizer for work with Nvidia Apex.

    Args:
        model: torch model
        optimizer: torch optimizer
        **engine_params: extra params for ``apex.amp.initialize``

    Returns:
        model and optimizer, wrapped with Nvidia Apex initialization
    """
    amp_params = get_fn_default_params(apex.amp.initialize, ["models", "optimizers"])
    amp_params["opt_level"] = "O0"
    for dp in engine_params:
        if dp in amp_params:
            amp_params[dp] = engine_params[dp]

    # NVIDIA apex support only:
    #  model: nn.Module or list of modules
    #  optimizer: None, torch.Optimizer or list of optimizers
    # while key-value is preferred in the `catalyst`.
    # So if model/optimizer is a dict, convert it to lists of keys
    # and values first, and then cast it back after apex initialization
    model_keys, optimizer_keys = None, None
    if isinstance(model, dict):
        model_keys, model = list(model.keys()), list(model.values())
    if isinstance(optimizer, dict):
        optimizer_keys = list(optimizer.keys())
        optimizer = list(optimizer.values())

    amp_result = apex.amp.initialize(model, optimizer, **amp_params)
    if optimizer is not None:
        model, optimizer = amp_result
    else:
        model = amp_result

    # convert model/optimizer back to dict if it needed
    if model_keys is not None:
        model = OrderedDict([(k, v) for k, v in zip(model_keys, model)])
    if optimizer_keys is not None:
        optimizers = [(k, v) for k, v in zip(optimizer_keys, optimizer)]
        optimizer = OrderedDict(optimizers)
    return model, optimizer


# taken form https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/components.py
def _patch_forward(model):
    input_caster_lambda = (
        lambda tensor: tensor.to(
            apex.amp._amp_state.opt_properties.options["cast_model_type"]
        )  # noqa: WPS437
        if tensor.is_floating_point()
        else tensor
    )
    output_caster_lambda = (
        lambda tensor: tensor.to(
            apex.amp._amp_state.opt_properties.options.get(
                "cast_model_outputs", torch.float32
            )  # noqa: WPS437
        )
        if tensor.is_floating_point()
        else tensor
    )

    def new_fwd(
        *args,
        old_fwd=model.forward,
        input_caster=input_caster_lambda,
        output_caster=output_caster_lambda,
        **kwargs,
    ):
        return apex.amp._initialize.applier(  # noqa: WPS437
            old_fwd(
                *apex.amp._initialize.applier(args, input_caster),  # noqa: WPS437
                **apex.amp._initialize.applier(kwargs, input_caster),  # noqa: WPS437
            ),
            output_caster,
        )

    model.forward = new_fwd
    return model


# taken form https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/components.py
# apex issue https://github.com/deepset-ai/FARM/issues/210
# solution: https://github.com/NVIDIA/apex/issues/503#issuecomment-566181771
def _wrap_into_data_parallel_with_apex(
    model: RunnerModel, optimizer: RunnerOptimizer, distributed_params: Dict
):
    if isinstance(model, nn.Module):
        model = nn.Sequential(model)
        model, optimizer = _initialize_apex(model, optimizer, **distributed_params)
        model = torch.nn.DataParallel(model[0])
        model = _patch_forward(model)
    elif isinstance(model, dict):
        model = {k: nn.Sequential(v) for k, v in model.items()}
        model, optimizer = _initialize_apex(model, optimizer, **distributed_params)
        model = {k: nn.DataParallel(v[0]) for k, v in model.items()}
        model = {k: _patch_forward(v) for k, v in model.items()}
    else:
        raise NotImplementedError()

    return model, optimizer


class APEXEngine(DeviceEngine):
    """Apex single training device engine.

    Args:
        device: use device, default is `"cuda"`.
        opt_level: optimization level, should be one of "O0", "O1", "O2", "O3" or "O4".

            - "O0" - no-op training
            - "O1" - mixed precision (FP16) training (default)
            - "O2" - "almost" mixed precision training
            - "O3" - another implementation of mixed precision training

            Details about levels can be found here:
            https://nvidia.github.io/apex/amp.html#opt-levels
        keep_batchnorm_fp32: To enhance precision and enable cudnn batchnorm
            (which improves performance),
            it’s often beneficial to keep batchnorm weights in FP32 even
            if the rest of the model is FP16.
        loss_scale: If loss_scale is a float value,
            use this value as the static (fixed) loss scale. If loss_scale is the string "dynamic",
            adaptively adjust the loss scale over time.
            Dynamic loss scale adjustments are performed by Amp automatically.
    """

    def __init__(
        self,
        device: str = "cuda",
        opt_level: str = "O1",
        keep_batchnorm_fp32: bool = None,
        loss_scale: Union[float, str] = None,
    ):
        """Init."""
        super().__init__(device)
        self.opt_level = opt_level
        self.keep_batchnorm_fp32 = keep_batchnorm_fp32
        self.loss_scale = loss_scale

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}',opt_level='{self.opt_level}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        # model
        model = model_fn()
        # model = _patch_forward(model)
        model = self.sync_device(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        # from official docs:
        #   https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        model, optimizer = _initialize_apex(
            model,
            optimizer,
            opt_level=self.opt_level,
            keep_batchnorm_fp32=self.keep_batchnorm_fp32,
            loss_scale=self.loss_scale,
        )

        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        """
        Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
        and some extra info ``**kwargs`` to torch-based checkpoint.

        Args:
            model: torch model
            criterion: torch criterion
            optimizer: torch optimizer
            scheduler: torch scheduler
            **kwargs: some extra info to pack

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
        """
        checkpoint = {"amp": amp.state_dict()}
        checkpoint = super().pack_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **checkpoint,
        )
        return checkpoint

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:
        """Load checkpoint from file and unpack the content to a model
        (if not None), criterion (if not None), optimizer (if not None),
        scheduler (if not None).

        Args:
            checkpoint: checkpoint to load
            model: model where should be updated state
            criterion: criterion where should be updated state
            optimizer: optimizer where should be updated state
            scheduler: scheduler where should be updated state
        """
        super().unpack_checkpoint(
            checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        # NOTE: propper way to load state, docs:
        #   https://nvidia.github.io/apex/amp.html#checkpointing
        if "amp" in checkpoint:
            amp.load_state_dict(checkpoint["amp"])


class DataParallelApexEngine(APEXEngine):
    """Apex multi-gpu training device engine."""

    def __init__(self, opt_level: str = "O1"):
        """Init."""
        super().__init__(f"cuda:{torch.cuda.current_device()}", opt_level)
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}',opt_level='{self.opt_level}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        model, optimizer = _wrap_into_data_parallel_with_apex(
            model, optimizer, distributed_params={"opt_level": self.opt_level}
        )

        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler


class DistributedDataParallelApexEngine(DistributedDataParallelEngine):
    """Distributed Apex MultiGPU training device engine.

    Args:
        address: process address to use (required for PyTorch backend), default is `"localhost"`.
        port: process port to listen (required for PyTorch backend), default is `"12345"`.
        backend: multiprocessing backend to use, default is `"nccl"`.
        world_size: number of processes.
        opt_level: optimization level, should be one of "O0", "O1", "O2", "O3" or "O4".

            - "O0" - no-op training
            - "O1" - mixed precision (FP16) training (default)
            - "O2" - "almost" mixed precision training
            - "O3" - another implementation of mixed precision training

            Details about levels can be found here:
            https://nvidia.github.io/apex/amp.html#opt-levels

        keep_batchnorm_fp32: To enhance precision and enable cudnn batchnorm
            (which improves performance),
            it’s often beneficial to keep batchnorm weights in FP32 even
            if the rest of the model is FP16.
        loss_scale: If loss_scale is a float value,
            use this value as the static (fixed) loss scale. If loss_scale is the string "dynamic",
            adaptively adjust the loss scale over time.
            Dynamic loss scale adjustments are performed by Amp automatically.
        delay_all_reduce: boolean flag for delayed all reduce
    """

    def __init__(
        self,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
        world_size: int = None,
        opt_level: str = "O1",
        keep_batchnorm_fp32: bool = None,
        loss_scale: Union[float, str] = None,
        delay_all_reduce: bool = True,
    ):
        """Init."""
        super().__init__()
        self.address = address
        self.port = port
        self.backend = backend
        self._rank = 0
        self._world_size = world_size or torch.cuda.device_count()
        self.device = None
        self.opt_level = opt_level
        self.delay_all_reduce = delay_all_reduce
        self.keep_batchnorm_fp32 = keep_batchnorm_fp32
        self.loss_scale = loss_scale

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, backend='{self.backend}', "
            f"rank={self._rank}, world_size={self._world_size}, "
            f"opt_level='{self.opt_level}')"
        )

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        # model, optimizer = _wrap_into_data_parallel_with_apex(
        #     model,
        #     optimizer,
        #     distributed_params=dict(
        #         opt_level=self.opt_level,
        #         keep_batchnorm_fp32=self.keep_batchnorm_fp32,
        #         loss_scale=self.loss_scale,
        #     ),
        # )
        # model = APEX_DDP(model, delay_allreduce=self.delay_all_reduce)
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=self.opt_level,
            keep_batchnorm_fp32=self.keep_batchnorm_fp32,
            loss_scale=self.loss_scale,
        )
        model = ApexDistributedDataParallel(model, delay_allreduce=self.delay_all_reduce)

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        """
        Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
        and some extra info ``**kwargs`` to torch-based checkpoint.

        Args:
            model: torch model
            criterion: torch criterion
            optimizer: torch optimizer
            scheduler: torch scheduler
            **kwargs: some extra info to pack

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
        """
        checkpoint = {"amp": amp.state_dict()}
        checkpoint = super().pack_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **checkpoint,
        )
        return checkpoint

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:
        """Load checkpoint from file and unpack the content to a model
        (if not None), criterion (if not None), optimizer (if not None),
        scheduler (if not None).

        Args:
            checkpoint: checkpoint to load
            model: model where should be updated state
            criterion: criterion where should be updated state
            optimizer: optimizer where should be updated state
            scheduler: scheduler where should be updated state
        """
        super().unpack_checkpoint(
            checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        # NOTE: propper way to load state, docs:
        #   https://nvidia.github.io/apex/amp.html#checkpointing
        if "amp" in checkpoint:
            amp.load_state_dict(checkpoint["amp"])


__all__ = ["APEXEngine", "DataParallelApexEngine", "DistributedDataParallelApexEngine"]
