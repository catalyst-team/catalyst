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
    :members:
    :undoc-members:
    :special-members: __init__, __call__


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
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

ListDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.ListDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

MergeDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.MergeDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

NumpyDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.NumpyDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

PathsDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.torch.PathsDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

Metric Learning Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MetricLearningTrainDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.metric_learning.MetricLearningTrainDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

QueryGalleryDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.dataset.metric_learning.QueryGalleryDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__


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
    :members:
    :exclude-members: __dict__, __module__
    :special-members:

BatchPrefetchLoaderWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.loader.BatchPrefetchLoaderWrapper
    :members:
    :exclude-members: __dict__, __module__
    :special-members:


Samplers
--------------------------------------

BalanceBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BalanceBatchSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

BalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.BalanceClassSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

DistributedSamplerWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DistributedSamplerWrapper
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

DynamicBalanceClassSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DynamicBalanceClassSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

DynamicLenBatchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.DynamicLenBatchSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__

MiniEpochSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.sampler.MiniEpochSampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__


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
    :members:
    :undoc-members:
    :special-members: __init__, __call__

ImageReader (CV)
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.reader.ImageReader
    :members:
    :undoc-members:
    :show-inheritance:

LambdaReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.LambdaReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

MaskReader (CV)
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.reader.MaskReader
    :members:
    :undoc-members:
    :show-inheritance:

ScalarReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.ScalarReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

ReaderCompose
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.reader.ReaderCompose
    :members:
    :undoc-members:
    :special-members: __init__, __call__


Mixins (CV)
~~~~~~~~~~~~~~~~~~~~~~~~

BlurMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.mixins.blur.BlurMixin
    :members:
    :undoc-members:
    :show-inheritance:

FlareMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.mixins.flare.FlareMixin
    :members:
    :undoc-members:
    :show-inheritance:

RotateMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.mixins.rotate.RotateMixin
    :members:
    :undoc-members:
    :show-inheritance:


Transforms (CV)
~~~~~~~~~~~~~~~~~~~~~~~~

Compose
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.torch.Compose
    :members:
    :undoc-members:
    :show-inheritance:

ImageToTensor
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.albumentations.ImageToTensor
    :members:
    :undoc-members:
    :show-inheritance:

Normalize
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.torch.Normalize
    :members:
    :undoc-members:
    :show-inheritance:

OneOfPerBatch
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.kornia.OneOfPerBatch
    :members:
    :undoc-members:
    :show-inheritance:

OneOfPerSample
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.kornia.OneOfPerSample
    :members:
    :undoc-members:
    :show-inheritance:

TensorToImage
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.albumentations.TensorToImage
    :members:
    :undoc-members:
    :show-inheritance:

ToTensor
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.transforms.torch.ToTensor
    :members:
    :undoc-members:
    :show-inheritance:


Datasets (CV)
~~~~~~~~~~~~~~~~

ImageFolderDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.cv.dataset.ImageFolderDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__


Datasets (NLP)
~~~~~~~~~~~~~~~~

LanguageModelingDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.nlp.dataset.language_modeling.LanguageModelingDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

TextClassificationDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.nlp.dataset.text_classification.TextClassificationDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__