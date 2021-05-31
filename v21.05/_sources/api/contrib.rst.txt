Contrib
================================================

.. toctree::
   :titlesonly:

.. contents::
   :local:

.. automodule:: catalyst.contrib
    :members:
    :undoc-members:


Datasets
--------------------

.. automodule:: catalyst.contrib.datasets
    :members:
    :undoc-members:
    :show-inheritance:

MNIST
~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.mnist.MNIST
    :members: __init__
    :undoc-members:
    :show-inheritance:

MovieLens
~~~~~~~~~~~~~~~~
.. autoclass:: catalyst.contrib.datasets.movielens.MovieLens
    :members: __init__
    :undoc-members:
    :show-inheritance:

Computer Vision
~~~~~~~~~~~~~~~~

Imagenette
""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagenette.Imagenette
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagenette160
"""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagenette.Imagenette160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagenette320
"""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagenette.Imagenette320
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang
"""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagewang.Imagewang
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang160
""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagewang.Imagewang160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewang320
""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagewang.Imagewang320
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof
"""""""""
.. autoclass:: catalyst.contrib.datasets.cv.imagewoof
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof160
""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.Imagewoof160
    :members: __init__
    :undoc-members:
    :show-inheritance:

Imagewoof320
""""""""""""
.. autoclass:: catalyst.contrib.datasets.cv.Imagewoof320
    :members: __init__
    :undoc-members:
    :show-inheritance:



NN
--------------------

Extensions for ``torch.nn``

Criterion
~~~~~~~~~~~~~~~~


CircleLoss
""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.circle.CircleLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

DiceLoss
"""""""""
.. autoclass:: catalyst.contrib.nn.criterion.dice.DiceLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalLossBinary
"""""""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.focal.FocalLossBinary
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalLossMultiClass
"""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.focal.FocalLossMultiClass
    :members: __init__
    :undoc-members:
    :show-inheritance:

IoULoss
"""""""""
.. autoclass:: catalyst.contrib.nn.criterion.iou.IoULoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

MarginLoss
""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.margin.MarginLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

TrevskyLoss
"""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.trevsky.TrevskyLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

FocalTrevskyLoss
""""""""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.trevsky.FocalTrevskyLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:

TripletMarginLossWithSampler
""""""""""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.criterion.triplet.TripletMarginLossWithSampler
    :members: __init__
    :undoc-members:
    :show-inheritance:

WingLoss
"""""""""
.. autoclass:: catalyst.contrib.nn.criterion.wing.WingLoss
    :members: __init__
    :undoc-members:
    :show-inheritance:


Modules
~~~~~~~~~~~~~~~~

ArcFace and SubCenterArcFace
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.arcface
    :members:
    :undoc-members:
    :show-inheritance:

Arc Margin Product
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.arcmargin
    :members:
    :undoc-members:
    :show-inheritance:

.. catalyst.contrib.nn.modules.common

CosFace and AdaCos
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.cosface
    :members:
    :undoc-members:
    :show-inheritance:

CurricularFace
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.curricularface
    :members:
    :undoc-members:
    :show-inheritance:

.. catalyst.contrib.nn.modules.lama
.. catalyst.contrib.nn.modules.pooling
.. catalyst.contrib.nn.modules.rms_norm


sSE
"""""""""
.. autoclass:: catalyst.contrib.nn.modules.se.sSE
    :members: __init__
    :undoc-members:
    :show-inheritance:

cSE
"""""""""
.. autoclass:: catalyst.contrib.nn.modules.se.cSE
    :members: __init__
    :undoc-members:
    :show-inheritance:

scSE
"""""""""
.. autoclass:: catalyst.contrib.nn.modules.se.scSE
    :members: __init__
    :undoc-members:
    :show-inheritance:


SoftMax
""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.softmax
    :members:
    :undoc-members:
    :show-inheritance:

Optimizers
~~~~~~~~~~~~~~~~

AdamP
"""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.adamp.AdamP
    :members: __init__
    :undoc-members:
    :show-inheritance:

Lamb
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.lamb.Lamb
    :members: __init__
    :undoc-members:
    :show-inheritance:

Lookahead
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.lookahead.Lookahead
    :members: __init__
    :undoc-members:
    :show-inheritance:

QHAdamW
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.qhadamw.QHAdamW
    :members: __init__
    :undoc-members:
    :show-inheritance:

RAdam
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.radam.RAdam
    :members: __init__
    :undoc-members:
    :show-inheritance:

Ralamb
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.ralamb.Ralamb
    :members: __init__
    :undoc-members:
    :show-inheritance:

SGDP
"""""""""""""
.. autoclass:: catalyst.contrib.nn.optimizers.sgdp.SGDP
    :members: __init__
    :undoc-members:
    :show-inheritance:


Schedulers
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.contrib.nn.schedulers
    :members:
    :undoc-members:
    :show-inheritance:

OneCycleLRWithWarmup
""""""""""""""""""""
.. autoclass:: catalyst.contrib.nn.schedulers.onecycle.OneCycleLRWithWarmup
    :members: __init__
    :undoc-members:
    :show-inheritance:

.. catalyst.contrib.models.cv.segmentation.unet
.. catalyst.contrib.models.cv.segmentation.linknet
.. catalyst.contrib.models.cv.segmentation.fpn
.. catalyst.contrib.models.cv.segmentation.psp

Scripts
--------------------

You can use contrib scripts with `catalyst-contrib` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-contrib tag2label --help

.. automodule:: catalyst.contrib.__main__
    :members:
    :exclude-members: build_parser, main
