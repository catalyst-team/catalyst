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
.. automodule:: catalyst.contrib.datasets.mnist
    :members:
    :undoc-members:
    :show-inheritance:

MovieLens
~~~~~~~~~~~~~~~~
.. automodule:: catalyst.contrib.datasets.movielens
    :members:
    :undoc-members:
    :show-inheritance:

Computer Vision
~~~~~~~~~~~~~~~~

Imagenette
""""""""""
.. automodule:: catalyst.contrib.datasets.cv.imagenette
    :members:
    :undoc-members:
    :show-inheritance:

Imagewoof
"""""""""
.. automodule:: catalyst.contrib.datasets.cv.imagewoof
    :members:
    :undoc-members:
    :show-inheritance:

Imagewang
"""""""""
.. automodule:: catalyst.contrib.datasets.cv.imagewang
    :members:
    :undoc-members:
    :show-inheritance:


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

Circle
"""""""""
.. automodule:: catalyst.contrib.nn.criterion.circle
    :members:
    :undoc-members:
    :show-inheritance:

Contrastive
"""""""""""
.. automodule:: catalyst.contrib.nn.criterion.contrastive
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

Common modules
""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.common
    :members:
    :undoc-members:
    :show-inheritance:

CosFace and AdaCos
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. automodule:: catalyst.contrib.nn.modules.cosface
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
.. automodule:: catalyst.contrib.nn.optimizers.adamp
    :members:
    :undoc-members:
    :show-inheritance:

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

SGDP
"""""""""""""
.. automodule:: catalyst.contrib.nn.optimizers.sgdp
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

Scripts
--------------------

You can use contrib scripts with `catalyst-contrib` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-contrib tag2label --help

.. automodule:: catalyst.contrib.__main__
    :members:
    :exclude-members: build_parser, main
