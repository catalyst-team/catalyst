Contrib
================================================

| Note: under development, best contrib modules will be placed here with docs and examples.
| If you would like to see your contribution here - please open a Pull Request or write us on `slack`_.
|

Catalyst contrib modules are supported in the code-as-a-documentation format.
If you are interested in the details - please, follow the code of the implementation.
If you are interested in contributing to the library - feel free to open a pull request.
For more information, please follow the `code for contrib-based extensions`_.

.. _`code for contrib-based extensions`: https://github.com/catalyst-team/catalyst/tree/master/catalyst/contrib
.. _`slack`: https://join.slack.com/t/catalyst-team-devs/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw


Data
--------------------

.. automodule:: catalyst.contrib.data
    :members:
    :undoc-members:
    :show-inheritance:

InBatchSamplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

InBatchTripletsSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler_inbatch.InBatchTripletsSampler
    :members: 
    :undoc-members:
    :show-inheritance:

AllTripletsSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler_inbatch.AllTripletsSampler
    :members: 
    :undoc-members:
    :show-inheritance:

HardTripletsSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler_inbatch.HardTripletsSampler
    :members: 
    :undoc-members:
    :show-inheritance:

HardClusterSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler_inbatch.HardClusterSampler
    :members: 
    :undoc-members:
    :show-inheritance: 

Samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BalanceBatchSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler.BalanceBatchSampler
    :members: 
    :undoc-members:
    :show-inheritance:

DynamicBalanceClassSampler
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.sampler.DynamicBalanceClassSampler
    :members: __init__
    :undoc-members:
    :show-inheritance:

Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compose
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.transforms.Compose
    :members: __init__
    :undoc-members:
    :show-inheritance:

ImageToTensor
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.transforms.ImageToTensor
    :members: __init__
    :undoc-members:
    :show-inheritance:

NormalizeImage
"""""""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.data.transforms.NormalizeImage
    :members: __init__
    :undoc-members:
    :show-inheritance:

Datasets
--------------------

.. automodule:: catalyst.contrib.datasets
    :members:
    :undoc-members:
    :show-inheritance:

CIFAR10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.cifar.CIFAR10
    :members: __init__
    :undoc-members:
    :show-inheritance:

CIFAR100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.cifar.CIFAR100
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagenette
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagenette.Imagenette
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagenette160
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagenette.Imagenette160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagenette320
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagenette.Imagenette320
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagewang.Imagewang
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang160
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagewang.Imagewang160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang320
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagewang.Imagewang320
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.imagewoof
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof160
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.Imagewoof160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof320
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.Imagewoof320
    :members: __init__
    :undoc-members:
    :show-inheritance:

MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.mnist.MNIST
    :members: __init__
    :undoc-members:
    :show-inheritance:

MovieLens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.movielens.MovieLens
    :members: __init__
    :undoc-members:
    :show-inheritance:

Layers
--------------------

AdaCos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.cosface.AdaCos
    :members:
    :undoc-members:
    :show-inheritance:

AMSoftmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.amsoftmax.AMSoftmax
    :members:
    :undoc-members:
    :show-inheritance:

ArcFace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.arcface.ArcFace
    :members:
    :undoc-members:
    :show-inheritance:

ArcMarginProduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.arcmargin.ArcMarginProduct
    :members:
    :undoc-members:
    :show-inheritance:

.. catalyst.contrib.layers.common

cSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.se.cSE
    :members: __init__
    :undoc-members:
    :show-inheritance:

CosFace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.cosface.CosFace
    :members:
    :undoc-members:
    :show-inheritance:

CurricularFace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.curricularface.CurricularFace
    :members:
    :undoc-members:
    :show-inheritance:

FactorizedLinear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.factorized.FactorizedLinear
    :members:
    :undoc-members:
    :show-inheritance:

.. catalyst.contrib.layers.lama
.. catalyst.contrib.layers.pooling
.. catalyst.contrib.layers.rms_norm

scSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.se.scSE
    :members: __init__
    :undoc-members:
    :show-inheritance:

SoftMax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.layers.softmax
    :members:
    :undoc-members:
    :show-inheritance:

sSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.se.sSE
    :members: __init__
    :undoc-members:
    :show-inheritance:

SubCenterArcFace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.layers.arcface.SubCenterArcFace
    :members:
    :undoc-members:
    :show-inheritance:

Losses
--------------------

AdaptiveHingeLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.AdaptiveHingeLoss
    :members:
    :undoc-members:
    :show-inheritance:

BarlowTwinsLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.contrastive.BarlowTwinsLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

BPRLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.BPRLoss
    :members:
    :undoc-members:
    :show-inheritance:

CircleLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.circle.CircleLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

DiceLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.dice.DiceLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalLossBinary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.focal.FocalLossBinary
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalLossMultiClass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.focal.FocalLossMultiClass
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalTrevskyLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.trevsky.FocalTrevskyLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

HingeLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.HingeLoss
    :members:
    :undoc-members:
    :show-inheritance:

HuberLossV0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.regression.HuberLossV0
    :members:
    :undoc-members:
    :show-inheritance:

IoULoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.iou.IoULoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

LogisticLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.LogisticLoss
    :members:
    :undoc-members:
    :show-inheritance:

NTXentLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.ntxent.NTXentLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

RocStarLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.RocStarLoss
    :members:
    :undoc-members:
    :show-inheritance:

RSquareLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.regression.RSquareLoss
    :members:
    :undoc-members:
    :show-inheritance:

SupervisedContrastiveLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.supervised_contrastive.SupervisedContrastiveLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

TrevskyLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.trevsky.TrevskyLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:


TripletMarginLossWithSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.triplet.TripletMarginLossWithSampler
    :members: __init__
    :undoc-members:
    :show-inheritance:

WARPLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.losses.recsys.WARPLoss
    :members:
    :undoc-members:
    :show-inheritance:


Optimizers
--------------------

AdamP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.adamp.AdamP
    :members: __init__
    :undoc-members:
    :show-inheritance:

Lamb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.lamb.Lamb
    :members: __init__
    :undoc-members:
    :show-inheritance:

Lookahead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.lookahead.Lookahead
    :members: __init__
    :undoc-members:
    :show-inheritance:

QHAdamW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.qhadamw.QHAdamW
    :members: __init__
    :undoc-members:
    :show-inheritance:

RAdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.radam.RAdam
    :members: __init__
    :undoc-members:
    :show-inheritance:

Ralamb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.ralamb.Ralamb
    :members: __init__
    :undoc-members:
    :show-inheritance:

SGDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.optimizers.sgdp.SGDP
    :members: __init__
    :undoc-members:

Schedulers
--------------------


OneCycleLRWithWarmup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.schedulers.onecycle.OneCycleLRWithWarmup
    :members: __init__
    :undoc-members:
    :show-inheritance:


