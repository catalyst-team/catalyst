Engines
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:

You could check engines overview under `examples/engines`_ section.

.. _`examples/engines`: https://github.com/catalyst-team/catalyst/tree/master/examples/engines

AMP
----------------------

AMPEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.amp.AMPEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DataParallelAMPEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.amp.DataParallelAMPEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DistributedDataParallelAMPEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.amp.DistributedDataParallelAMPEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:


Apex
----------------------

APEXEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.apex.APEXEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DataParallelApexEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.apex.DataParallelApexEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DistributedDataParallelApexEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.apex.DistributedDataParallelApexEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:


DeepSpeed
----------------------

DistributedDataParallelDeepSpeedEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.deepspeed.DistributedDataParallelDeepSpeedEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:


FairScale
----------------------

PipelineParallelFairScaleEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.fairscale.PipelineParallelFairScaleEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

SharedDataParallelFairScaleEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.fairscale.SharedDataParallelFairScaleEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

SharedDataParallelFairScaleAMPEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.fairscale.SharedDataParallelFairScaleAMPEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

FullySharedDataParallelFairScaleEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.fairscale.FullySharedDataParallelFairScaleEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:


Torch
----------------------

DeviceEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.DeviceEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DataParallelEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.DataParallelEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

DistributedDataParallelEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.DistributedDataParallelEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

