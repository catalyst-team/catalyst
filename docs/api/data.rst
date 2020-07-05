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
--------------------

You can use scripts typing `catalyst-data` in your terminal.
For example:

.. code-block:: bash

    $ catalyst-data tag2label --help

.. automodule:: catalyst.data.__main__
    :members:
    :exclude-members: build_parser, main


Augmentor
--------------------

.. automodule:: catalyst.data.augmentor
    :members:
    :undoc-members:


Collate Functions
--------------------

.. automodule:: catalyst.data.collate_fn
    :members:
    :undoc-members:
    :special-members: __init__, __call__


Dataset
--------------------

.. automodule:: catalyst.data.dataset
    :show-inheritance:
    :members:
    :special-members: __getitem__, __len__


In-batch Samplers
--------------------

.. automodule:: catalyst.data.sampler_inbatch
    :members:
    :undoc-members:
    :special-members: __iter__, __len__


Loader
--------------------

.. automodule:: catalyst.data.loader
    :show-inheritance:
    :members:
    :special-members:


Reader
--------------------

Readers are the abstraction for your dataset. They can open an elem from the dataset and transform it to data, needed by your network.
For example open image by path, or read string and tokenize it.

.. automodule:: catalyst.data.reader
    :members:
    :undoc-members:
    :special-members: __init__, __call__


Sampler
--------------------

.. automodule:: catalyst.data.sampler
    :members:
    :undoc-members:
    :special-members: __iter__, __len__
