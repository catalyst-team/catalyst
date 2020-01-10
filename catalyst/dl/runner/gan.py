import torch
from collections import OrderedDict
from typing import Any, Mapping, Dict  # isort: skip
from catalyst.dl.callbacks import (
    PhaseBatchWrapperCallback,
    PhaseManagerCallback,
    CriterionAggregatorCallback,
)
from catalyst.dl import (
    Runner,
    Callback,
    OptimizerCallback,
    CriterionCallback,
)
from catalyst.dl.experiment import GanExperiment
from catalyst.utils.typing import (
    Model,
    Device,
    Optimizer,
    Criterion,
    DataLoader,
    Dataset
)


class GanRunner(Runner):
    """
    Runner for experiments with GANs
    """

    _default_experiment = GanExperiment
    TRAIN_PARAMS = {
        "main_metric": "loss_g",
        "minimize_metric": True,
        "batch_consistant_metrics": False,
        "discriminator_train_phase": "discriminator_train",
        "generator_train_phase": "generator_train",
    }

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
            assert hasattr(self.state, key) \
                   and getattr(self.state, key) is not None

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

    def validate_models(self, models: Dict[str, Model]) -> None:
        assert isinstance(
            models, dict
        ), "models must be of Dict[str, torch.nn.Module] type"
        for key, model in models.items():
            assert isinstance(
                model, Model
            ), f"model {key} must be torch.nn.Module type"
        discriminator_assert_message = \
            f"models must have discriminator Module with {self.discriminator_key} key"  # noqa: E501
        assert self.discriminator_key in models, discriminator_assert_message
        assert isinstance(
            models[self.discriminator_key], torch.nn.Module
        ), discriminator_assert_message
        generator_assert_message = \
            f"models must have generator Module with {self.generator_key} key"  # noqa: E501
        assert self.generator_key in models, generator_assert_message
        assert isinstance(
            models[self.generator_key], torch.nn.Module
        ), generator_assert_message

    def prepare_state_params(
        self, num_epochs: int, logdir: str, additional_state_params
    ) -> Dict[str, Any]:
        state_params = {
            "num_epochs": num_epochs,
            "logdir": logdir,
            **self.TRAIN_PARAMS,
        }
        state_params.update(additional_state_params)
        return state_params

    @staticmethod
    def prepare_experiment_params(
        models,
        criterion,
        optimizers,
        state_params,
        callbacks,
        datasets,
        loaders,
        train_stage_name="train",
    ) -> Dict:
        experiment_params = {
            "models": {train_stage_name: models},
            "criterions": {train_stage_name: criterion},
            "optimizers": {train_stage_name: optimizers},
            "stages": [train_stage_name],
            "state_params": {train_stage_name: state_params},
            "callbacks": {train_stage_name: callbacks},
            "datasets": {train_stage_name: datasets},
            "loaders": {train_stage_name: loaders},
        }
        return experiment_params

    def prepare_callbacks(
        self, callbacks, discriminator_phase_num, generator_phase_num
    ) -> OrderedDict:
        """
            Catalyst can't take several optimizer callbacks for generator
            and discriminator and process them raw
            Therefore, phase manager, criterion callbacks and wrappers
            will be created to manage training order
        """
        assert isinstance(
            callbacks, dict
        ), "callbacks should be of dict[str, Callback] type"
        discriminator_phase = self.TRAIN_PARAMS["discriminator_train_phase"]
        generator_phase = self.TRAIN_PARAMS["generator_train_phase"]
        callbacks["phase_manager"] = PhaseManagerCallback(
            train_phases=OrderedDict(
                [
                    (discriminator_phase, discriminator_phase_num),
                    (generator_phase, generator_phase_num),
                ]
            ),
            valid_mode="all",
        )
        callbacks["loss_g"] = PhaseBatchWrapperCallback(
            base_callback=CriterionCallback(
                input_key="real_targets",
                output_key="fake_logits",
                prefix="loss_g",
            ),
            active_phases=[generator_phase],
        )
        callbacks["loss_d_real"] = PhaseBatchWrapperCallback(
            base_callback=CriterionCallback(
                input_key="real_targets",
                output_key="real_logits",
                prefix="loss_d_real",
            ),
            active_phases=[discriminator_phase],
        )
        callbacks["loss_d_fake"] = PhaseBatchWrapperCallback(
            base_callback=CriterionCallback(
                input_key="fake_targets",
                output_key="fake_logits",
                prefix="loss_d_fake",
            ),
            active_phases=[discriminator_phase],
        )
        callbacks["loss_d"] = PhaseBatchWrapperCallback(
            base_callback=CriterionAggregatorCallback(
                loss_keys=["loss_d_real", "loss_d_fake"],
                loss_aggregate_fn="mean",
                prefix="loss_d",
            ),
            active_phases=[discriminator_phase],
        )
        callbacks["optim_g"] = PhaseBatchWrapperCallback(
            base_callback=OptimizerCallback(
                loss_key="loss_g", optimizer_key=self.generator_key,
            ),
            active_phases=[generator_phase],
        )
        callbacks["optim_d"] = PhaseBatchWrapperCallback(
            base_callback=OptimizerCallback(
                loss_key="loss_d", optimizer_key=self.discriminator_key,
            ),
            active_phases=[discriminator_phase],
        )
        return OrderedDict(callbacks)

    def train(
        self,
        models: Dict[str, Model],
        criterion: Criterion,
        optimizers: Dict[str, Optimizer],
        datasets: Dict[str, Dataset],
        loaders: Dict[str, DataLoader],
        callbacks: Dict[str, Callback] = {},
        discriminator_phase_num: int = 1,
        generator_phase_num: int = 1,
        num_epochs: int = 1,
        additional_state_params: Dict[str, Any] = {},
        logdir: str = "./logs",
        verbose: bool = False,
        check: bool = False,
    ):
        # Validate models type and its items
        self.validate_models(models)
        # Check for optimizers
        assert (
            self.generator_key in optimizers.keys()
        ), f"optimizers must have optimizer with {self.generator_key} key"
        assert (
            self.discriminator_key in optimizers.keys()
        ), f"optimizers must have optimizer with {self.discriminator_key} key"
        # Init from parameters and set default state params
        state_params = self.prepare_state_params(
            num_epochs, logdir, additional_state_params
        )
        callbacks = self.prepare_callbacks(
            callbacks, discriminator_phase_num, generator_phase_num
        )
        # Prepare parameters of one-stage experiment
        experiment_stage_params = self.prepare_experiment_params(
            models=models,
            criterion=criterion,
            optimizers=optimizers,
            state_params=state_params,
            callbacks=callbacks,
            datasets=datasets,
            loaders=loaders,
        )
        # Initialize and run experiment
        experiment = self._default_experiment(
            **experiment_stage_params, logdir=logdir, verbose=verbose
        )
        self.run_experiment(experiment=experiment, check=check)


__all__ = ["GanRunner"]
