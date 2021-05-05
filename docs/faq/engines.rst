Engines
==============================================================================

Catalyst has engines - it's an abstraction over a different hardwares like
gpus or tpu cores. Engine manages all hardware dependent operations
like initializations/loading/saving components in DDP setup, casting values
in APEX/AMP or loss scaling etc.

Based on device availability there are 3 groups:

- ``DeviceEngine``, ``AMPEngine``, ``APEXEngine`` -
    run experiments using single gpu or even a cpu

- ``DataParallelEngine``, ``DataParallelAMPEngine``, ``DataParallelApexEngine`` -
    run experiments using multiple gpus using single process

- ``DistributedDataParallelEngine``, ``DistributedDataParallelAMPEngine``, ``DistributedDataParallelApexEngine`` -
    run experiments using multiple gpus using multiple processes


Based on hardware features there are also 3 groups:

- ``DeviceEngine``, ``DataParallelEngine``, ``DistributedDataParallelEngine`` -
    pure pytorch without anything else (fp32 training)

- ``AMPEngine``, ``DataParallelAMPEngine``, ``DistributedDataParallelAMPEngine`` -
    training with pytorch automatic mixed precision package
    (some operations in fp32 and other in fp16),
    more information you can find here: https://pytorch.org/docs/stable/amp.html


- ``APEXEngine``, ``DataParallelApexEngine``, ``DistributedDataParallelApexEngine`` -
    training with nvidia's automatic mixed precision package
    more information you can find here: https://github.com/NVIDIA/apex


To create a new engine you need to inherit from IEngine and implement some methods:

.. code-block:: python

    class MyEngine(IEngine):
        
        @property
        def rank(self):  
            # ddp property, should return rank number of a process
            # if ddp is not required then return -1
            pass

        @property
        def world_size(self) -> int:
            # ddp property, should return number of processes in a process group
            # if ddp is not required then return 1
            pass


        @property
        def is_master_process(self) -> bool:
            # ddp property, should return True if current process is a master process
            #   otherwise should return False
            # if ddp is not required then return True
            return True

        @property
        def is_worker_process(self) -> bool:
            # ddp property, should return True if current process isn't a master process
            #   otherwise should return False
            # if ddp is not required then return False
            return False
        
        def sync_device(self, tensor_or_module: Any) -> Any:
            # move tensor or module to a device specified in engine
            #   the same as `tensor_or_module.to(device)`
            pass

        def sync_tensor(self, tensor: Any, mode: str) -> Any:
            # ddp property, synchronized tensor across processes
            #   modes - sum, mean, concatenate
            pass

        def init_components(self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None):
            # initialize experiment components and move them to a device specified in engine
            pass

        def deinit_components(self):
            # ddp property, destroy processes
            pass

        def zero_grad(self, loss, model, optimizer) -> None:
            # abstraction over model.zero_grad()
            pass

        def backward_loss(self, loss, model, optimizer) -> None:
            # abstraction over loss.backward()
            # for example AMP scales loss in this method
            pass

        def optimizer_step(self, loss, model, optimizer) -> None:
            # abstraction over optimizer.step()
            # for example AMP scales optimizer and perform an optimization step
            pass

        def pack_checkpoint(
            self,
            model: Model = None,
            criterion: Criterion = None,
            optimizer: Optimizer = None,
            scheduler: Scheduler = None,
            **kwargs,
        ) -> Dict:
            # method to collect components states for later use in checkpoint saving
            pass

        def unpack_checkpoint(
            self,
            checkpoint: Dict,
            model: Model = None,
            criterion: Criterion = None,
            optimizer: Optimizer = None,
            scheduler: Scheduler = None,
            **kwargs,
        ) -> None:
            # method to setup components state dicts from a checkpoint dict
            pass

        def save_checkpoint(self, checkpoint: Dict, path: str) -> None:
            # method for saving checkpoints
            pass

        def load_checkpoint(self, path: str) -> Dict:
            # method for loading checkpoints
            pass

        def autocast(self, *args, **kwargs):
            # amp method for automatic casting values to a fp16 during the forward propagation
            # if casting is not required then should return null context
            # in general this method used like this:
            #   with engine.autocast():
            #       output = model(batch)
            return nullcontext()

