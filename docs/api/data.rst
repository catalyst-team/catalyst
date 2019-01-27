Data
============

Data subpackage has useful scripts and classes for data preprocessing

.. automodule:: catalyst.data
    :members:
    :undoc-members:

Scripts
---------

.. automodule:: catalyst.data.__main__
    :members:

Reader
--------

Readers are the abstraction for your dataset. They can open an elem from the dataset and transform it to data, needed by your network.
For example open image by path, or read string and tokenize it.

.. automodule:: catalyst.data.reader
    :members:
    :undoc-members:
    :special-members: __call__, __init__


Dataset
---------

.. automodule:: catalyst.data.dataset
    :members:
    :undoc-members:

Sampler
------------

.. automodule:: catalyst.data.sampler
    :members:
    :undoc-members:


Collate Functions
--------------------

.. automodule:: catalyst.data.collate_fn
    :members:
    :undoc-members:


Mixins
----------

.. automodule:: catalyst.data.mixin
    :members:
    :undoc-members:


Augmentor
------------

.. automodule:: catalyst.data.augmentor
    :members:
    :undoc-members:
