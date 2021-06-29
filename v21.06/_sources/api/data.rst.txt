Data
================================================

Data subpackage has data preprocessers and dataloader abstractions.

.. toctree::
   :titlesonly:

.. contents::
   :local:


Main
----------------------
.. automodule:: catalyst.data
    :members:
    :show-inheritance:

Collate Functions
--------------------------------------

FilteringCollateFn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.collate_fn.FilteringCollateFn
    :members: __init__
    :undoc-members:
    :special-members:


Dataset
--------------------------------------

.. automodule:: catalyst.data.dataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

PyTorch Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DatasetFromSampler
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.DatasetFromSampler
    :members: __init__
    :show-inheritance:

ListDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.ListDataset
    :members: __init__
    :show-inheritance:

MergeDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.MergeDataset
    :members: __init__
    :show-inheritance:

NumpyDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.NumpyDataset
    :members: __init__
    :show-inheritance:

PathsDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.PathsDataset
    :members: __init__
    :show-inheritance:

Metric Learning Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MetricLearningTrainDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.metric_learning.MetricLearningTrainDataset
    :members: __init__, get_labels
    :show-inheritance:

QueryGalleryDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.metric_learning.QueryGalleryDataset
    :members: __init__, query_size, gallery_size
    :show-inheritance:


In-batch Samplers
--------------------------------------

IInbatchTripletSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.IInbatchTripletSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

InBatchTripletsSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.InBatchTripletsSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

AllTripletsSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.AllTripletsSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

HardTripletsSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.HardTripletsSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

HardClusterSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.HardClusterSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

Loader
--------------------------------------

BatchLimitLoaderWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.loader.BatchLimitLoaderWrapper
    :members: __init__
    :exclude-members:
    :special-members:

BatchPrefetchLoaderWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.loader.BatchPrefetchLoaderWrapper
    :members: __init__
    :exclude-members:
    :special-members:


Samplers
--------------------------------------

BalanceBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BalanceBatchSampler
    :members: __init__
    :undoc-members:
    :special-members:

BalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BalanceClassSampler
    :members: __init__
    :undoc-members:
    :special-members:

DistributedSamplerWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DistributedSamplerWrapper
    :members: __init__
    :undoc-members:
    :special-members:

DynamicBalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DynamicBalanceClassSampler
    :members: __init__
    :undoc-members:
    :special-members:

DynamicLenBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DynamicLenBatchSampler
    :members: __init__
    :undoc-members:
    :special-members:

MiniEpochSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.MiniEpochSampler
    :members: __init__
    :undoc-members:
    :special-members:

Transforms
--------------------------------------

Compose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.transforms.Compose
    :members: __init__
    :undoc-members:
    :show-inheritance:

Normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.transforms.Normalize
    :members: __init__
    :undoc-members:
    :show-inheritance:

ToTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.transforms.ToTensor
    :members: __init__
    :undoc-members:
    :show-inheritance:

Contrib
----------------------
.. automodule:: catalyst.contrib.data
    :members:
    :show-inheritance:

Augmentors
~~~~~~~~~~~~~~~~

Augmentor
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.augmentor.Augmentor
    :members:
    :undoc-members:

AugmentorCompose
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.augmentor.AugmentorCompose
    :members:
    :undoc-members:

AugmentorKeys
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.augmentor.AugmentorKeys
    :members:
    :undoc-members:

Readers
~~~~~~~~~~~~~~~~

Readers are the abstraction for your dataset. They can open an elem from the dataset and transform it to data, needed by your network.
For example open image by path, or read string and tokenize it.

IReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.IReader
    :members: __init__
    :undoc-members:
    :special-members:

ImageReader (CV)
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.reader.ImageReader
    :members: __init__
    :undoc-members:
    :show-inheritance:

LambdaReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.LambdaReader
    :members: __init__
    :undoc-members:
    :special-members:

MaskReader (CV)
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.reader.MaskReader
    :members: __init__
    :undoc-members:
    :show-inheritance:

NiftiReader (Nifti)
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.nifti.reader.NiftiReader
    :members: __init__
    :undoc-members:
    :show-inheritance:

ScalarReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.ScalarReader
    :members: __init__
    :undoc-members:
    :special-members:

ReaderCompose
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.ReaderCompose
    :members: __init__
    :undoc-members:
    :special-members:


Datasets (CV)
~~~~~~~~~~~~~~~~

ImageFolderDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.dataset.ImageFolderDataset
    :show-inheritance:
    :members: __init__
    :special-members:
