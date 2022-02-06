Engines
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:

You could check engines overview under `examples/engines`_ section.

.. _`examples/engines`: https://github.com/catalyst-team/catalyst/tree/master/examples/engines

Torch
----------------------

CPUEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.CPUEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

GPUEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.GPUEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:

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

DistributedXLAEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.engines.torch.DistributedXLAEngine
    :exclude-members: __init__
    :undoc-members:
    :show-inheritance:
