from collections import OrderedDict
from typing import Any, Mapping, Dict, List, Union  # isort: skip

import torch

from catalyst.dl import (
    Runner,
    Callback,
)
from catalyst.dl.experiment import GanExperiment
from catalyst.utils.typing import (
    Model,
    Device,
    Optimizer,
    Criterion,
    DataLoader,
)


class GanRunner(Runner):
    """
    Runner for experiments with GANs
    """

    _default_experiment = GanExperiment

    def __init__(
        self,
        model: Dict[str, Model] = None,
        device: Device = None,
        features_key: str = "features",
        generator_key: str = "generator",
        discriminator_key: str = "discriminator",
    ):
        """
        Args:
            model (Model): dict with two models: discriminator and generator
            device (Device): Torch device to run model on
            features_key (str): Key to extract features from batch
            generator_key (str): Key in model dict for generator model
            discriminator_key (str): Key in model dict for discriminator model
        """
        super().__init__(model=model, device=device)
        self.features_key = features_key
        self.generator_key = generator_key
        self.discriminator_key = discriminator_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            batch = {self.features_key: batch[0]}
        return super()._batch2device(batch, device)

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage)
        self.generator = self.model[self.generator_key]
        self.discriminator = self.model[self.discriminator_key]
        for key in [
            "noise_dim",
            "discriminator_train_phase",
            "generator_train_phase",
        ]:
            assert (
                hasattr(self.state, key)
                and getattr(self.state, key) is not None
            )

    def forward(self, batch):
        real_features = batch[self.features_key]
        batch_size = real_features.shape[0]
        real_targets = torch.ones((batch_size, 1), device=self.device)
        fake_targets = torch.zeros((batch_size, 1), device=self.device)
        self.state.input["real_targets"] = real_targets
        self.state.input["fake_targets"] = fake_targets
        z = torch.randn((batch_size, self.state.noise_dim), device=self.device)
        if (
            self.state.phase is None
            or self.state.phase == self.state.discriminator_train_phase
        ):
            # (None for validation mode)
            fake_features = self.generator(z)
            fake_logits = self.discriminator(fake_features.detach())
            real_logits = self.discriminator(real_features)
            # --> d_loss
            # (fake logits + FAKE targets) + (real logits + real targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits,
                "real_logits": real_logits,
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z)
            fake_logits = self.discriminator(fake_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits,
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")

    def validate_models(self, model: Dict[str, Model]) -> None:
        """Validate model dict to have generator and discriminator model"""
        assert isinstance(
            model, dict
        ), "model must be of Dict[str, torch.nn.Module] type"
        for key, submodel in model.items():
            assert isinstance(
                submodel, Model
            ), f'model item with key value "{key}" must be Model type'
        discriminator_assert_message = f"model must have discriminator Model with {self.discriminator_key} key"  # noqa: E501
        assert self.discriminator_key in model, discriminator_assert_message
        assert isinstance(
            model[self.discriminator_key], torch.nn.Module
        ), discriminator_assert_message
        generator_assert_message = f"model must have generator Model with {self.generator_key} key"  # noqa: E501
        assert self.generator_key in model, generator_assert_message
        assert isinstance(
            model[self.generator_key], torch.nn.Module
        ), generator_assert_message

    def validate_optimizers(self, optimizer: Dict[str, Optimizer]) -> None:
        """
            Validate model dict to have optimizer
            for generator and discriminator model
        """
        assert (
            self.generator_key in optimizer.keys()
        ), f"optimizer dict must have optimizer with {self.generator_key} key"  # noqa: E501
        assert (
            self.discriminator_key in optimizer.keys()
        ), f"optimizer dict must have optimizer with {self.discriminator_key} key"  # noqa: E501

    def train(
        self,
        model: Union[Model, Dict[str, Model]],
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "OrderedDict[str, Callback]" = None,
        logdir: str = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        num_epochs: int = 1,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        distributed_params: Dict = None,
        monitoring_params: Dict = None,
        initial_seed: int = 42,
        phase2callbacks: Dict[str, List[str]] = None,
        check: bool = False,
    ) -> None:
        """
        Args:
            model (Model or Dict[str, Model]): models,
                usually generator and discriminator
            loaders (dict): dictionary containing one or several
                ``torch.utils.data.DataLoader`` for training and validation
            callbacks (List[catalyst.dl.Callback]): list of callbacks
            logdir (str): path to output directory
            stage (str): current stage
            criterion (Criterion): criterion function
            optimizer (Optimizer): optimizer
            scheduler (Scheduler): scheduler
            num_epochs (int): number of experiment's epochs
            valid_loader (str): loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric (str): the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric (bool): flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose (bool): ff true, it displays the status of the training
                to the console.
            state_kwargs (dict): additional state params to ``RunnerState``
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            distributed_params (dict): dictionary with the parameters
                for distributed and FP16 method
            monitoring_params (dict): dict with the parameters
                for monitoring services
            initial_seed (int): experiment's initial seed value
            phase2callbacks (dict): dictionary with lists of callback names
                which should be wrapped for appropriate phase
                for example: {"generator_train": "loss_g", "optim_g"}
                "loss_g" and "optim_g" callbacks from callbacks dict
                will be wrapped for "generator_train" phase
                in wrap_callbacks method
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only)
        """
        # Validate model type and its items
        self.validate_models(model)
        # Check for optimizers
        self.validate_optimizers(optimizer)
        # Check phase parameters in state_kwargs
        consistent_metrics_param_key = "batch_consistant_metrics"
        if consistent_metrics_param_key not in state_kwargs:
            state_kwargs[consistent_metrics_param_key] = False
        # @TODO: self.validate_state_kwargs(state_kwargs)
        # Initialize and run experiment
        experiment = self._default_experiment(
            model=model,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            verbose=verbose,
            state_kwargs=state_kwargs,
            checkpoint_data=checkpoint_data,
            distributed_params=distributed_params,
            monitoring_params=monitoring_params,
            initial_seed=initial_seed,
            phase2callbacks=phase2callbacks,
        )
        self.run_experiment(experiment=experiment, check=check)


__all__ = ["GanRunner"]
