Engines
==============================================================================

Catalyst has engines - it's an abstraction over different hardware like CPU,
GPUs or TPU cores. Engine manages all hardware-dependent operations
like initializations, loading checkpoints, saving experiment components in DDP setup,
casting values in AMP or loss scaling, etc.

Based on device availability there are 3 groups:

- ``CPUEngine``, ``GPUEngine`` - run experiments using one device like CPU or GPU.


- ``DataParallelEngine`` -
    run experiments using multiple GPUs within one process


- ``DistributedDataParallelEngine``, ``DistributedXLAEngine`` -
    run experiments using multiple GPUs/TPUs within several processes

