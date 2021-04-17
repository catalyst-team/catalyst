Dataflow
==============================================================================

Base dataflow
----------------------------------------------------
Catalyst uses the "key-value is all you need" approach.
In other words, the output of your Dataset/Dataloader should be a key-value (python `dict`).

Example of dataflow:

.. code-block:: python

    class MyDataset:

        def __get_item__(self, index):
            ...
            return {"features": np.ndarray, "targets": np.ndarray}

    class MyModel:

        def forward(self, features):
            ...
            return logits

    class MyRunner:

        def handle_batch(self, batch):
            # on this step we also have self.batch = batch = {"features": ..., "targets": ...}
            logits = self.model(batch["features"])
            self.batch.update({"logits": logits})
            # this is useful for other components of the pipeline

    loader = MyDataset()
    model = MyModel()
    runner = MyRunner()
    runner.train(...)

.. note::

    ``SupervisedRunner`` has data preprocessing features to transform
    tuple/list-based data into key-value.

Such approach is easily extensible for any number of features, targets and
very convenient to read, thanks to "automatic naming documentation" - keys for the values:

.. code-block:: python

    class MyDataset:

        def __get_item__(self, index):
            ...
            return {"features": np.ndarray, "extra_features": np.ndarray, "targets": np.ndarray}

    class MyModel:

        def forward(self, features, extra_features):
            ...
            return logits

    class MyRunner:

        def handle_batch(self, batch):
            # on this step we also have
            # self.batch = batch = {"features": ..., "extra_features": ...,"targets": ...}
            logits = self.model(batch["features"], batch["extra_features"])
            self.batch.update()"logits": logits})
            # this is useful for other components of the pipeline

    loader = MyDataset()
    model = MyModel()
    runner = MyRunner()
    runner.train(...)

Moreover, if some of the features are not required anymore -
you don't have to rewrite your code:

.. code-block:: python

    class MyDataset:

        def __get_item__(self, index):
            ...
            return {"features": np.ndarray, "extra_features": np.ndarray, "targets": np.ndarray}

    class MyModel:

        def forward(self, features):
            ...
            return logits

    class MyRunner:

        def handle_batch(self, batch):
            # on this step we also have
            # self.batch = batch = {"features": ..., "extra_features": ...,"targets": ...}
            logits = self.model(batch["features"])
            self.batch.update({"logits": logits})
            # this is useful for other components of the pipeline

    loader = MyDataset()
    model = MyModel()
    runner = MyRunner()
    runner.train(...)


Key-value storage also can be used to store the datasets/loaders for the experiment.
In this case we also need to use ``OrderedDict`` to ensure correct epoch handling -
that your model will firstly train on some ``train`` dataset
and only then will be evaluated on some ``valid`` dataset:

.. code-block:: python

    train_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    loaders = OrderedDict("train": train_loader, "valid": valid_loader)
    model = MyModel()
    runner = MyRunner()
    runner.train(model=model, loaders=loaders, ...)

Catalyst uses the following "automatic naming documentation" for loader keys handling:

- if ``loader_key`` starts with "train" - is's train datasoure, we need to run forward and backward passes on it.
- if ``loader_key`` starts with "valid" - is's validation datasoure, we need to run forward, but not the backward pass on it.
- if ``loader_key`` starts with "infer" - is's inference datasoure, we need to run forward, but not the backward pass on it.

Multiple datasources
----------------------------------------------------
Thanks to key-value approach,
it's possible to handle any number of datasets/loader
without code changes or tricks with Datasets concatenation, etc:

.. code-block:: python

    train_loader = MyDataset(...)
    train2_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    valid2_loader = MyDataset(...)
    loaders = OrderedDict(
        "train": train_loader,
        "train2": train2_loader,
        "valid": valid_loader,
        "valid2": valid2_loader,
    )
    model = MyModel()
    runner = MyRunner()
    runner.train(model=model, loaders=loaders, ...)

What is even more interesting, you could also do something like:

.. code-block:: python

    train_loader = MyDataset(...)
    train2_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    valid2_loader = MyDataset(...)
    loaders = OrderedDict(
        "train": train_loader,
        "valid": valid_loader,
        "train2": train2_loader,
        "valid2": valid2_loader,

    )
    model = MyModel()
    runner = MyRunner()
    runner.train(model=model, loaders=loaders, ...)

Once again, it's also valid to do something like:

.. code-block:: python

    train_loader = MyDataset(...)
    train2_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    valid2_loader = MyDataset(...)
    loaders = OrderedDict(
        "train": concat_datasets(train_loader, train2_loader),
        "valid": concat_datasets(valid_loader, valid2_loader),
    )
    model = MyModel()
    runner = MyRunner()
    runner.train(model=model, loaders=loaders)


Loader for model selection
----------------------------------------------------
In case of multiple loaders, you could select one of them for model selection
with ``valid_loader``, ``valid_metric`` and ``minimize_valid_metric`` params in the ``runner.train``.
For example, to use ``valid2`` loaders as your
model selection one you could do the following:

.. code-block:: python

    train_loader = MyDataset(...)
    train2_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    valid2_loader = MyDataset(...)
    loaders = OrderedDict(
        "train": train_loader,
        "train2": train2_loader,
        "valid": valid_loader,
        "valid2": valid2_loader,
    )
    model = MyModel()
    criterion = ...
    optimizer = ...
    runner = MyRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        valid_loader="valid2",
        valid_metric="loss",
        minimize_valid_metric=True
    )

.. note::

    By default, Catalyst doesn't suppose to use
    ``valid_loader``, ``valid_metric`` and ``minimize_valid_metric``
    for model selection.


Metric for model selection
----------------------------------------------------
Suppose, you are using set of different metrics in your pipeline:

.. code-block:: python

    class MyRunner:

        def handle_batch(self, batch):
            # on this step we also have self.batch = batch = {"features": ..., "targets": ...}
            logits = self.model(batch["features"])
            self.output = {"logits": logits}
            # this is useful for other components of the pipeline

    loaders = {"train": ..., "valid": ...}
    model = ...
    criterion = ...
    optimizer = ...
    runner = MyRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=[
            AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3))
        ]
    )

You could select one for model selection with ``valid_loader``, ``valid_metric``
and ``minimize_valid_metric`` params in the ``runner.train``.
For example, to use ``accuracy01`` metric as your model selection one you could do the following:

.. code-block:: python

    class MyRunner:

        def handle_batch(self, batch):
            # on this step we also have self.batch = batch = {"features": ..., "targets": ...}
            logits = self.model(batch["features"])
            self.output = {"logits": logits}
            # this is useful for other components of the pipeline

    loaders = {"train": ..., "valid": ...}
    model = ...
    criterion = ...
    optimizer = ...
    runner = MyRunner()
    # as far as we would like to maximize our model accuracy...
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        valid_loader="valid",
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        callbacks=[
            AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3))
        ]
    )

.. note::

    By default, Catalyst doesn't suppose to use
    ``valid_loader``, ``valid_metric`` and ``minimize_valid_metric``
    for model selection.

Use part of the data
----------------------------------------------------
If you would like to use only some part of your data from the loader
(for example, you would like to overfit for one small portion of the data to check your pipeline),
you could use ``BatchLimitLoaderWrapper``:

.. code-block:: python

    train_loader = BatchLimitLoaderWrapper(MyDataset(...), num_batches=1)
    valid_loader = MyDataset(...)
    loaders = OrderedDict("train": train_loader, "valid": valid_loader)
    model = MyModel()
    runner = MyRunner()
    runner.train(model=model, loaders=loaders)

As a more user-friendly approach with ``runner.train``:

.. code-block:: python

    train_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    loaders = OrderedDict("train": train_loader, "valid": valid_loader)
    model = MyModel()
    runner = MyRunner()
    # here we overfit for one batch per loader
    runner.train(model=model, loaders=loaders, overfit=True)

And more convenient and customizable way:

.. code-block:: python

    train_loader = MyDataset(...)
    valid_loader = MyDataset(...)
    loaders = OrderedDict("train": train_loader, "valid": valid_loader)
    model = MyModel()
    runner = MyRunner()
    # here we overfit for 10 batches in `train` loader
    # and half of the `valid` loader
    runner.train(
        model=model,
        loaders=loaders,
        callbacks=[dl.BatchOverfitCallback(train=10, valid=0.5)]
    )

----

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
