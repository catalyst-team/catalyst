Data
============

Data subpackage has data preprocessers and dataloader abstractions.

.. automodule:: catalyst.data
    :members:
    :undoc-members:

Scripts
---------

You can use scripts typing `catalyst-data` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-data tag2label --help

.. automodule:: catalyst.data.__main__
    :members:

Reader
--------

Readers are the abstraction for your dataset. They can open an elem from the dataset and transform it to data, needed by your network.
For example open image by path, or read string and tokenize it.


.. currentmodule:: catalyst.data.reader

.. autoclass:: BaseReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

.. autoclass:: LambdaReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

.. autoclass:: ScalarReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

.. autoclass:: ImageReader
    :members:
    :undoc-members:
    :special-members: __init__, __call__

.. autoclass:: ReaderCompose
    :members:
    :undoc-members:
    :special-members: __init__, __call__


Dataset
---------

.. automodule:: catalyst.data.dataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__

Sampler
------------

.. automodule:: catalyst.data.sampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__


Collate Functions
--------------------

.. automodule:: catalyst.data.collate_fn
    :members:
    :undoc-members:
    :special-members: __init__, __call__


Mixins
----------

.. automodule:: catalyst.data.mixin
    :members:
    :undoc-members:


Augmentor
------------

Legacy classes for augmentations.
For modern Catalyst use `albumentations`_.

.. _albumentations: https://github.com/albu/albumentations

.. automodule:: catalyst.data.augmentor
    :members:
    :undoc-members:
