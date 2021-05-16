Engines
==============================================================================

Catalyst has engines - it's an abstraction over different hardware like CPU,
GPUs or TPU cores. Engine manages all hardware-dependent operations
like initializations, loading checkpoints, saving experiment components in DDP setup,
casting values in APEX/AMP or loss scaling, etc.

Based on device availability there are 3 groups:

- ``DeviceEngine``, ``AMPEngine``, ``APEXEngine`` -
    run experiments using one device like CPU or GPU


- ``DataParallelEngine``, ``DataParallelAMPEngine``, ``DataParallelApexEngine`` -
    run experiments using multiple GPUs using one process


- ``DistributedDataParallelEngine``, ``DistributedDataParallelAMPEngine``, ``DistributedDataParallelApexEngine`` -
    run experiments using multiple GPUs using multiple processes


Based on hardware features there are also 3 groups:

- ``DeviceEngine``, ``DataParallelEngine``, ``DistributedDataParallelEngine`` -
    pure PyTorch without anything else (fp32 training)


- ``AMPEngine``, ``DataParallelAMPEngine``, ``DistributedDataParallelAMPEngine`` -
    training with PyTorch automatic mixed precision package
    (some operations in fp32 and other in fp16),
    more information you can find from `PyTorch docs <https://pytorch.org/docs/stable/amp.html>`_.


- ``APEXEngine``, ``DataParallelApexEngine``, ``DistributedDataParallelApexEngine`` -
    training with Nvidia's automatic mixed precision package
    more information you can find from `Nvidia APEX docs <https://github.com/NVIDIA/apex>`_.


The easiest way to implement a new Engine is to inherit from an already existed one.

For example, if you want to add functionality to a device engine you can inherit
from ``DeviceEngine`` and overload some methods.

This approach was used during the dev process and you can see that
``DataParallelEngine`` inherits from ``DeviceEngine`` and extends it.

The same for ``DistributedDataParallelApexEngine`` and
``DistributedDataParallelAMPEngine`` - they both inherited from ``DistributedDataParallelEngine``.


If you want to do it in a hard way then need to inherit from ``IEngine`` and implement the required methods:

- ``rank`` - it's a DDP **property** and should return a rank number of a process.
    If engine is used outside DDP then should always return ``-1``.

    .. code-block:: python

        @property
        def rank(self):
            return -1


- ``world_size`` - it's a DDP **property** and should return a total number of processes inside a process group.
    If engine is used outside DDP then should always return ``1``.

    .. code-block:: python
        
        @property
        def world_size(self):
            return 1


- ``is_master_process`` - it's a DDP **property** and it's a simple indicator for a master process.
    If engine is used outside DDP then should always return ``True``.

    .. code-block:: python

        @property
        def is_master_process(self):
            return True


- ``is_worker_process`` - it's a DDP **property** and it's a simple indicator that the current process isn't a master process.
    If engine is used outside DDP then should always return ``False``.

    .. code-block:: python

        @property
        def is_worker_process(self):
            return False


- ``sync_device`` - function to move PyTorch tensor or module to a device specified in engine.
    In general, it wraps `tensor_or_module.to(device)` function.

    .. code-block:: python
        
        def sync_device(self, tensor_or_module):
            return tensor_or_module.to(self.device)


- ``sync_tensor`` - it's a DDP function to synchronize tensor across processes and perform sum/mean/all_gather operation.
    If engine is used outside DDP then should always return the same tensor.

    .. code-block:: python

        def sync_tensor(self, tensor, mode=None):
            return tensor


- ``init_components`` - function to initialize model, criterion, optimizer, scheduler on a device specified in engine.

    .. code-block:: python

        def init_components(
            self,
            model_fn=None,
            criterion_fn=None,
            optimizer_fn=None,
            scheduler_fn=None,
        ):
            model = model_fn()
            model = self.sync_device(model)

            criterion = criterion_fn()
            criterion = self.sync_device(criterion)

            optimizer = optimizer_fn()
            optimizer = self.sync_device(optimizer)

            scheduler = scheduler_fn()
            scheduler = self.sync_device(scheduler)
            return model, criterion, optimizer, scheduler


- ``deinit_components`` - it's a DDP function to destroy process components.
    If engine is used outside DDP then should always do nothing.

    .. code-block:: python

        # ddp example
        def deinit_components(self):
            dist.barrier()
            dist.destroy_process_group()


- ``zero_grad`` - abstraction over``model.zero_grad()``.

    .. code-block:: python

        def zero_grad(self, loss, model, optimizer):
            model.zero_grad()


- ``backward_loss`` - abstraction over ``loss.backward()``.

    .. code-block:: python

        def backward_loss(self, loss, model, optimizer):
            loss.backward()


- ``optimizer_step`` - abstraction over ``optimizer.step()``.

    .. code-block:: python

        def optimizer_step(self, loss, model, optimizer):
            optimizer.step()


- ``pack_checkpoint`` - function to collect components state dicts for later save to checkpoint file.

    .. code-block:: python

        def pack_checkpoint(
            self,
            model=None,
            criterion=None,
            optimizer=None,
            scheduler=None,
            **kwargs
        ):
            content = {}
            if model is not None:
                content["model_state_dict"] = model.state_dict()
            if criterion is not None:
                content["criterion_state_dict"] = criterion.state_dict()
            if optimizer is not None:
                content["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                content["scheduler_state_dict"] = scheduler.state_dict()
            return content


- ``unpack_checkpoint`` - function to setup components from checkpoint state dict.

    .. code-block:: python
        
        def unpack_checkpoint(
            self,
            checkpoint,
            model=None,
            criterion=None,
            optimizer=None,
            scheduler=None,
            **kwargs,
        ):
            state_dicts = ("model", "criterion", "optimizer", "scheduler"):
            parts = (model, criterion, optimizer, scheduler)
            for state_dict, part in zip(state_dicts, parts):
                if f"{state_dict}_state_dict" in checkpoint and part is not None:
                    part.load_state_dict(checkpoint[f"{state_dict}_state_dict"])

- ``save_checkpoint`` - function to save checkpoint dict to file.
    It is an abstraction over ``torch.save``

    .. code-block:: python

        def save_checkpoint(self, checkpoint, path):
            torch.save(checkpoint, path)


- ``load_checkpoint`` - function to load checkpoint dict from file.
    It is an abstraction over ``torch.load``

    .. code-block:: python
        
        def load_checkpoint(self, path):
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            return checkpoint


- ``autocast`` - it's a AMP function for automatic casting values to a FP16 during the forward propagation.
    It wraps forward of a model like this:

    .. code-block:: python

        with engine.autocast():
            output= model(batch)

    If engine is used outside AMP then always should yield nothing.

    .. code-block:: python

        def autocast(self, *args, **kwargs):

            import contextlib

            @contextlib.contextmanager
            def nullcontext(to_yield):
                yield to_yield

            return nullcontext()
