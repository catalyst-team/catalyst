Contrib
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:


Data
--------------------

.. automodule:: catalyst.contrib.data
    :members:
    :undoc-members:
    :show-inheritance:

Transforms
~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.data.transforms
    :members:
    :undoc-members:
    :show-inheritance:

Computer Vision
~~~~~~~~~~~~~~~~

Mixins
"""""""""""""
.. automodule:: catalyst.contrib.data.cv.mixins.blur
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.contrib.data.cv.mixins.flare
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.contrib.data.cv.mixins.rotate
    :members:
    :undoc-members:
    :show-inheritance:

Transforms
"""""""""""""
.. automodule:: catalyst.contrib.data.cv.transforms.tensor
    :members:
    :undoc-members:
    :show-inheritance:

Reader
"""""""""""""
.. automodule:: catalyst.contrib.data.cv.reader
    :members:
    :undoc-members:
    :show-inheritance:


Natural Language Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset
"""""""""""""

.. automodule:: catalyst.contrib.data.nlp.dataset.language_modeling
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.contrib.data.nlp.dataset.text_classification
    :members:
    :undoc-members:
    :show-inheritance:


Datasets
--------------------

.. automodule:: catalyst.contrib.datasets
    :members:
    :undoc-members:
    :show-inheritance:

MNIST
~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.datasets.mnist
    :members:
    :undoc-members:
    :show-inheritance:


DL
--------------------


Callbacks
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.contrib.dl.callbacks
    :members:
    :undoc-members:
    :show-inheritance:

AlchemyLogger
"""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.alchemy_logger
    :members:
    :undoc-members:
    :show-inheritance:

CutmixCallback
""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.cutmix_callback
    :members:
    :undoc-members:
    :show-inheritance:

GradNormLogger
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.gradnorm_logger
    :members:
    :undoc-members:
    :show-inheritance:

KNNMetricCallback
"""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.knn_metric
    :members:
    :undoc-members:
    :show-inheritance:

InferMaskCallback
"""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.mask_inference
    :members:
    :undoc-members:
    :show-inheritance:

NeptuneLogger
"""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.neptune_logger
    :members:
    :undoc-members:
    :show-inheritance:

PeriodicLoaderCallback
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.periodic_loader_callback
    :members:
    :undoc-members:
    :show-inheritance:

PerplexityMetricCallback
""""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.perplexity_metric
    :members:
    :undoc-members:
    :show-inheritance:

TelegramLogger
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.telegram_logger
    :members:
    :undoc-members:
    :show-inheritance:

TracerCallback
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.tracer_callback
    :members:
    :undoc-members:
    :show-inheritance:

VisdomLogger
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.visdom_logger
    :members:
    :undoc-members:
    :show-inheritance:

WandbLogger
""""""""""""""""""""""
.. automodule:: catalyst.contrib.dl.callbacks.wandb_logger
    :members:
    :undoc-members:
    :show-inheritance:


Models
--------------------

NN
--------------------

Extensions for ``torch.nn``

Criterion
~~~~~~~~~~~~~~~~

Cross entropy
"""""""""""""
.. automodule:: catalyst.contrib.nn.criterion.ce
    :members:
    :undoc-members:
    :show-inheritance:

Contrastive
"""""""""""
.. automodule:: catalyst.contrib.nn.criterion.contrastive
    :members:
    :undoc-members:
    :show-inheritance:

Circle
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.circle
    :members:
    :undoc-members:
    :show-inheritance:

Dice
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.dice
    :members:
    :undoc-members:
    :show-inheritance:

Focal
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.focal
    :members:
    :undoc-members:
    :show-inheritance:

GAN
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.gan
    :members:
    :undoc-members:
    :show-inheritance:

Huber
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.huber
    :members:
    :undoc-members:
    :show-inheritance:

IOU
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.iou
    :members:
    :undoc-members:
    :show-inheritance:

Lovasz
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.lovasz
    :members:
    :undoc-members:
    :show-inheritance:

Margin
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.margin
    :members:
    :undoc-members:
    :show-inheritance:

Triplet
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.triplet
    :members:
    :undoc-members:
    :show-inheritance:

Wing
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.wing
    :members:
    :undoc-members:
    :show-inheritance:


Modules
~~~~~~~~~~~~~~~~

Common modules
""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.common
    :members:
    :undoc-members:
    :show-inheritance:

Last-Mean-Average-Attention (LAMA)-Pooling
""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.lama
    :members:
    :undoc-members:
    :show-inheritance:

Pooling
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.pooling
    :members:
    :undoc-members:
    :show-inheritance:

RMSNorm
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.rms_norm
    :members:
    :undoc-members:
    :show-inheritance:

SqueezeAndExcitation
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.se
    :members:
    :undoc-members:
    :show-inheritance:



Optimizers
~~~~~~~~~~~~~~~~

Lamb
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.lamb
    :members:
    :undoc-members:
    :show-inheritance:

Lookahead
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.lookahead
    :members:
    :undoc-members:
    :show-inheritance:

QHAdamW
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.qhadamw
    :members:
    :undoc-members:
    :show-inheritance:

RAdam
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.radam
    :members:
    :undoc-members:
    :show-inheritance:

Ralamb
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.ralamb
    :members:
    :undoc-members:
    :show-inheritance:


Schedulers
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.contrib.nn.schedulers.base
    :members:
    :undoc-members:
    :show-inheritance:

OneCycleLRWithWarmup
""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.schedulers.onecycle
    :members:
    :undoc-members:
    :show-inheritance:


Models
--------------------

Segmentation
~~~~~~~~~~~~~~~~

Unet
""""""""""""""""
.. automodule:: catalyst.contrib.models.cv.segmentation.unet
    :members:
    :undoc-members:
    :show-inheritance:

Linknet
""""""""""""""""
.. automodule:: catalyst.contrib.models.cv.segmentation.linknet
    :members:
    :undoc-members:
    :show-inheritance:

FPNnet
""""""""""""""""
.. automodule:: catalyst.contrib.models.cv.segmentation.fpn
    :members:
    :undoc-members:
    :show-inheritance:

PSPnet
""""""""""""""""
.. automodule:: catalyst.contrib.models.cv.segmentation.psp
    :members:
    :undoc-members:
    :show-inheritance:


Registry
--------------------

.. automodule:: catalyst.contrib.registry
    :members:
    :undoc-members:
    :show-inheritance:


Tools
------------------------

Tensorboard
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.tools.tensorboard
    :members:
    :undoc-members:
    :show-inheritance:


Utilities
------------------------

Argparse
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.argparse
    :members:
    :undoc-members:
    :show-inheritance:

Compression
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.compression
    :members:
    :undoc-members:
    :show-inheritance:

Confusion Matrix
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.confusion_matrix
    :members:
    :undoc-members:
    :show-inheritance:

Dataset
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.dataset
    :members:
    :undoc-members:
    :show-inheritance:

Misc
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.misc
    :members:
    :undoc-members:
    :show-inheritance:

Pandas
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.pandas
    :members:
    :undoc-members:
    :show-inheritance:

Parallel
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.parallel
    :members:
    :undoc-members:
    :show-inheritance:

Plotly
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.plotly
    :members:
    :undoc-members:
    :show-inheritance:

Serialization
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.serialization
    :members:
    :undoc-members:
    :show-inheritance:

Visualization
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.visualization
    :members:
    :undoc-members:
    :show-inheritance:


Computer Vision utilities
-------------------------

Image
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.cv.image
    :members:
    :undoc-members:
    :show-inheritance:

Tensor
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.cv.tensor
    :members:
    :undoc-members:
    :show-inheritance:


Natural Language Processing utilities
-------------------------------------

Text
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.utils.nlp.text
    :members:
    :undoc-members:
    :show-inheritance:
