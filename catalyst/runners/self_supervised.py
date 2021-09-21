from typing import Any, Mapping

from catalyst.core.runner import IRunner


class ISelfSupervisedRunner(IRunner):
    """IRunner for experiments with contrastive model.

    Args:
        input_key: key in ``runner.batch`` dict mapping for model input
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output
        augemention_prefix: key for ``runner.batch`` to sample augumentions
        projection_prefix: key for ``runner.batch`` to store model projection
        embedding_prefix: key for `runner.batch`` to store model embeddings

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.contrastive.ContrastiveRunner`

    .. note::
        ISelfSupervisedRunner contains only the logic with batch handling.


    ISelfSupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {"aug1": tensor, "aug2": tensor, ...}
        _, proj1 = model(batch["aug1"])
        _, proj2 = model(batch["aug2"])
        loss = criterion(proj1, proj2)
        batch_metrics["loss_key"] = loss

    Examples:

    .. code-block:: python

        # 1. loader and transforms

        transforms = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                torchvision.transforms.RandomCrop((28, 28)),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )
        mnist = MNIST("./logdir", train=True, download=True, transform=None)
        contrastive_mnist = ContrastiveDataset(mnist, transforms=transforms)

        train_loader = torch.utils.data.DataLoader(contrastive_mnist, batch_size=BATCH_SIZE)

        # 2. model and optimizer
        encoder = MnistSimpleNet(out_features=16)
        projection_head = nn.Sequential(
            nn.Linear(16, 16, bias=False), nn.ReLU(inplace=True), nn.Linear(16, 16, bias=True)
        )

        class ContrastiveModel(torch.nn.Module):
            def __init__(self, model, encoder):
                super(ContrastiveModel, self).__init__()
                self.model = model
                self.encoder = encoder

            def forward(self, x):
                emb = self.encoder(x)
                projection = self.model(emb)
                return emb, projection

        model = ContrastiveModel(model=projection_head, encoder=encoder)

        optimizer = Adam(model.parameters(), lr=LR)

        # 3. criterion with triplets sampling
        criterion = NTXentLoss(tau=0.1)

        callbacks = [
            dl.ControlFlowCallback(
                dl.CriterionCallback(
                    input_key="projection_left", target_key="projection_right", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.SklearnModelCallback(
                feature_key="embedding_left",
                target_key="target",
                train_loader="train",
                valid_loaders="valid",
                model_fn=RandomForestClassifier,
                predict_method="predict_proba",
                predict_key="sklearn_predict",
                random_state=RANDOM_STATE,
                n_estimators=10,
            ),
            dl.ControlFlowCallback(
                dl.AccuracyCallback(
                    target_key="target", input_key="sklearn_predict", topk_args=(1, 3)
                ),
                loaders="valid",
            ),
        ]

        runner = dl.ContrastiveRunner()

        logdir = "./logdir"
        runner.train(
            model=model,
            engine=engine or dl.DeviceEngine(device),
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": train_loader},
            verbose=True,
            logdir=logdir,
            valid_loader="train",
            valid_metric="loss",
            minimize_valid_metric=True,
            num_epochs=10,
        )

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples

    """

    def __init__(
        self,
        input_key: str = "features",
        target_key: str = "target",
        loss_key: str = "loss",
        augemention_prefix: str = "augment",
        projection_prefix: str = "projection",
        embedding_prefix: str = "embedding",
    ):
        """Init."""
        IRunner.__init__(self)

        self._target_key = target_key
        self._loss_key = loss_key
        self._projection_prefix = projection_prefix
        self._augemention_prefix = augemention_prefix
        self._embedding_prefix = embedding_prefix
        self._input_key = input_key

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) in [3, 4]

            if len(batch) == 4:
                batch = {
                    self._input_key: batch[0],
                    f"{self._augemention_prefix}_left": batch[1],
                    f"{self._augemention_prefix}_right": batch[2],
                    self._target_key: batch[3],
                }
            elif len(batch) == 3:
                batch = {
                    self._input_key: batch[0],
                    f"{self._augemention_prefix}_left": batch[1],
                    f"{self._augemention_prefix}_right": batch[2],
                }

        return batch

    def _process_input(self, batch: Mapping[str, Any], **kwargs):
        embedding1, projection1 = self.model(batch[f"{self._augemention_prefix}_left"], **kwargs)
        embedding2, projection2 = self.model(batch[f"{self._augemention_prefix}_right"], **kwargs)
        origin_embeddings, projection_origin = self.model(batch[self._input_key], **kwargs)
        batch = {
            **batch,
            f"{self._projection_prefix}_left": projection1,
            f"{self._projection_prefix}_right": projection2,
            f"{self._projection_prefix}_origin": projection_origin,
            f"{self._embedding_prefix}_left": embedding1,
            f"{self._embedding_prefix}_right": embedding2,
            f"{self._embedding_prefix}_origin": origin_embeddings,
        }

        return batch

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
        self.batch = self._process_batch(self.batch)
        super().on_batch_start(runner)

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner.
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model

        Returns:
            dict with model output batch
        """
        return self._process_input(batch, **kwargs)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch: dictionary with data batches from DataLoader.
        """
        self.batch = {**batch, **self.forward(batch)}


__all__ = ["ISelfSupervisedRunner"]
