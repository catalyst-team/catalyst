Quickstart 101
==============================================================================
**In this quickstart, we'll show you how to organize your PyTorch code with Catalyst.**

Catalyst goals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- flexibility, keeping the PyTorch simplicity, but removing the boilerplate code.
- readability by decoupling the experiment run.
- reproducibility.
- scalability to any hardware without code changes.
- extensibility for pipeline customization.

Step 1 - Install packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install using `pip package`_:

.. code:: bash

   pip install -U catalyst

.. _`pip package`: https://pypi.org/project/catalyst/

Step 2 - Make python imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl, utils
    from catalyst.contrib.datasets import MNIST

Step 3 - Write PyTorch code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's define **what** we would like to run:

.. code-block:: python

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False), batch_size=32),
    }

Step 4 - Accelerate it with Catalyst
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's define **how** we would like to handle the data (in pure PyTorch):

.. code-block:: python

    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.engine.device))

        def handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            logits = self.model(x)
            self.batch = {"features": x, "targets": y, "logits": logits}

Step 5 - Train the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's **train** and **evaluate** your model (`supported metrics`_) with a few lines of code.

.. _`supported metrics`: https://catalyst-team.github.io/catalyst/api/metrics.html

.. code-block:: python

    runner = CustomRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=5,
        verbose=True,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
            dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
            dl.BackwardCallback(metric_key="loss"),
            dl.OptimizerCallback(metric_key="loss"),
            dl.CheckpointCallback(
                "./logs", loader_key="valid", metric_key="loss", minimize=True, topk=3
            ),
        ]
    )

    # model evaluation
    metrics = runner.evaluate_loader(
        loader=loaders["valid"],
        callbacks=[dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3, 5))],
    )

Step 6 - Make predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You could easily use your custom logic for model inference on batch
or loader thanks to ``runner.predict_batch`` and ``runner.predict_loader`` methods.

.. code-block:: python

    # model batch inference
    features_batch = next(iter(loaders["valid"]))[0]
    prediction_batch = runner.predict_batch(features_batch)
    # model loader inference
    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10

Step 7 - Prepare for development stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, you could use a large number of model post-processing utils for production use cases.

.. code-block:: python

    model = runner.model.cpu()
    batch = next(iter(loaders["valid"]))[0]
    # model tracing
    utils.trace_model(model=model, batch=batch)
    # model quantization
    utils.quantize_model(model=model)
    # model pruning
    utils.prune_model(model=model, pruning_fn="l1_unstructured", amount=0.8)
    # onnx export
    utils.onnx_export(model=model, batch=batch, file="./logs/mnist.onnx", verbose=True)
