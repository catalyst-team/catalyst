Data
================================================

Data subpackage has data preprocessers and dataloader abstractions.

.. toctree::
   :titlesonly:

.. contents::
   :local:

Dataset
--------------------------------------

DatasetFromSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.dataset.DatasetFromSampler
    :exclude-members: __init__
    :show-inheritance:

SelfSupervisedDatasetWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.dataset.SelfSupervisedDatasetWrapper
    :exclude-members: __init__
    :show-inheritance:

Loader
--------------------------------------

BatchLimitLoaderWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.loader.BatchLimitLoaderWrapper
    :exclude-members: __init__
    :special-members:

BatchPrefetchLoaderWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.loader.BatchPrefetchLoaderWrapper
    :exclude-members: __init__
    :special-members:


Samplers
--------------------------------------

BalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BalanceClassSampler
    :exclude-members: __init__
    :special-members:

BatchBalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BatchBalanceClassSampler
    :exclude-members: __init__
    :special-members:

DistributedSamplerWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DistributedSamplerWrapper
    :exclude-members: __init__
    :special-members:

DynamicBalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DynamicBalanceClassSampler
    :exclude-members: __init__
    :special-members:

MiniEpochSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.MiniEpochSampler
    :exclude-members: __init__
    :special-members:
