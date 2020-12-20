Data
================================================

Data subpackage has data preprocessers and dataloader abstractions.

.. toctree::
   :titlesonly:

.. contents::
   :local:


.. automodule:: catalyst.data
    :members:
    :undoc-members:

Scripts
--------------------------------------

You can use scripts typing `catalyst-data` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-data tag2label --help

.. automodule:: catalyst.data.__main__
    :members:
    :exclude-members: build_parser, main


Augmentors
--------------------------------------

Augmentor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.augmentor.Augmentor
    :members:
    :undoc-members:

AugmentorCompose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.augmentor.AugmentorCompose
    :members:
    :undoc-members:

AugmentorKeys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.augmentor.AugmentorKeys
    :members:
    :undoc-members:


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


Readers
--------------------------------------

Readers are the abstraction for your dataset. They can open an elem from the dataset and transform it to data, needed by your network.
For example open image by path, or read string and tokenize it.

ReaderSpec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.reader.ReaderSpec
    :members:
    :undoc-members:
    :special-members: __init__, __call__

ScalarReader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.reader.ScalarReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

LambdaReader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.reader.LambdaReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

ReaderCompose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.data.reader.ReaderCompose
    :members:
    :undoc-members:
    :special-members: __init__, __call__


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


Computer Vision Extensions
--------------------------------------

Dataset
~~~~~~~~~~~~~~~~~~~~~~~~

ImageFolderDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.dataset.ImageFolderDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

Mixins
~~~~~~~~~~~~~~~~~~~~~~~~

BlurMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.mixins.blur.BlurMixin
    :members:
    :undoc-members:
    :show-inheritance:

FlareMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.mixins.flare.FlareMixin
    :members:
    :undoc-members:
    :show-inheritance:

RotateMixin
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.mixins.rotate.RotateMixin
    :members:
    :undoc-members:
    :show-inheritance:

Readers
~~~~~~~~~~~~~~~~~~~~~~~~

ImageReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.reader.ImageReader
    :members:
    :undoc-members:
    :show-inheritance:

MaskReader
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.reader.MaskReader
    :members:
    :undoc-members:
    :show-inheritance:

Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

TensorToImage
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.albumentations.TensorToImage
    :members:
    :undoc-members:
    :show-inheritance:

ImageToTensor
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.albumentations.ImageToTensor
    :members:
    :undoc-members:
    :show-inheritance:

Compose
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.torch.Compose
    :members:
    :undoc-members:
    :show-inheritance:

Normalize
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.torch.Normalize
    :members:
    :undoc-members:
    :show-inheritance:

ToTensor
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.torch.ToTensor
    :members:
    :undoc-members:
    :show-inheritance:

OneOfPerBatch
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.kornia.OneOfPerBatch
    :members:
    :undoc-members:
    :show-inheritance:

OneOfPerSample
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.cv.transforms.kornia.OneOfPerSample
    :members:
    :undoc-members:
    :show-inheritance:


Natural Language Processing Extensions
--------------------------------------

.. automodule:: catalyst.data.nlp
    :members:
    :undoc-members:
    :special-members:

Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LanguageModelingDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.nlp.dataset.language_modeling.LanguageModelingDataset
    :members:
    :undoc-members:
    :show-inheritance:

TextClassificationDataset
""""""""""""""""""""""""""
.. autoclass:: catalyst.data.nlp.dataset.text_classification.TextClassificationDataset
    :members:
    :undoc-members:
    :show-inheritance:
