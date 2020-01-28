from typing import (  # isort:skip
    Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
)

from catalyst.dl import Runner
from catalyst.utils.tools.typing import Device, Model


class MultiPhaseRunner(Runner):
    """
    Base Runner with multiple phases
    """
    def __init__(
        self,
        model: Union[Model, Dict[str, Model]] = None,
        device: Device = None,
        input_batch_keys: List[str] = None,
        registered_phases: Tuple[Tuple[str, Union[str, Callable]], ...] = None
    ):
        """

        :param model:
        :param device:
        :param input_batch_keys: list of strings of keys for batch elements,
            e.g. input_batch_keys = ["features", "targets"] and your
            DataLoader returns 2 tensors (images and targets)
            when state.input will be
            {"features": batch[0], "targets": batch[1]}
        :param registered_phases:
            Tuple of pairs (phase_name, phase_forward_function)
            phase_forward_function's may be also str, in that case Runner
            should have method with same name, which will be called
        """
        super().__init__(model, device)

        self.input_batch_keys = input_batch_keys or []

        self.registered_phases = dict()
        for phase_name, phase_batch_forward_fn in registered_phases:
            if not (isinstance(phase_name, str) or phase_name is None):
                raise ValueError(
                    f"phase '{phase_name}' of type '{type(phase_name)}' "
                    f"not supported, must be str of None"
                )
            if phase_name in self.registered_phases:
                raise ValueError(f"phase '{phase_name}' already registered")
            if isinstance(phase_batch_forward_fn, str):
                assert hasattr(self, phase_batch_forward_fn)
                phase_batch_forward_fn = getattr(self, phase_batch_forward_fn)
            assert isinstance(
                phase_batch_forward_fn, Callable
            ), "must be callable"
            self.registered_phases[phase_name] = phase_batch_forward_fn

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            assert len(batch) >= len(self.input_batch_keys)
            batch = {
                key: value
                for key, value in zip(self.input_batch_keys, batch)
            }
        return super()._batch2device(batch, device)

    def forward(self, batch, **kwargs):
        """Forward call"""
        if self.state.phase not in self.registered_phases:
            raise ValueError(f"Unknown phase: '{self.state.phase}'")

        return self.registered_phases[self.state.phase]()


class GanRunner(MultiPhaseRunner):
    """
    Runner with logic for single-generator single-discriminator GAN training

    Various conditioning types, penalties and regularization (such as WGAN-GP)
    can be easily derived from this class
    """
    def __init__(
        self,
        model: Union[Model, Dict[str, Model]] = None,
        device: Device = None,
        input_batch_keys: Optional[List[str]] = None,
        # input keys
        data_input_key: str = "data",
        class_input_key: str = "class_targets",
        noise_input_key: str = "noise",
        # output keys
        fake_logits_output_key: str = "fake_logits",
        real_logits_output_key: str = "real_logits",
        fake_data_output_key: str = "fake_data",
        # condition_keys
        fake_condition_keys: List[str] = None,
        real_condition_keys: List[str] = None,
        # phases:
        generator_train_phase: str = "generator_train",
        discriminator_train_phase: str = "discriminator_train",
        # model keys:
        generator_model_key: str = "generator",
        discriminator_model_key: str = "discriminator"
    ):
        """

        :param model:
        :param device:
        :param input_batch_keys: list of strings of keys for batch elements,
            e.g. input_batch_keys = ["features", "targets"] and
            your DataLoader returns 2 tensors (images and targets)
            when state.input will be
            {"features": batch[0], "targets": batch[1]}

        INPUT KEYS:
        :param data_input_key: real distribution to fit
        :param class_input_key: labels for real distribution
        :param noise_input_key: noise

        OUTPUT KEYS:
        :param fake_logits_output_key:  prediction scores of discriminator for
            fake data
        :param real_logits_output_key: prediction scores of discriminator for
            real data
        :param fake_data_output_key: generated data

        CONDITIONS:
        :param fake_condition_keys: list of all conditional inputs of
            discriminator (fake data conditions)
            (appear in same order as in generator model forward() call)
        :param real_condition_keys: list of all conditional inputs of
            discriminator (real data conditions)
            (appear in same order as in generator model forward() call)
        Note: THIS RUNNER SUPPORTS ONLY EQUALLY CONDITIONED generator and
            discriminator (i.e. if generator is conditioned on 3 variables,
            discriminator must be conditioned on same 3 variables)

        PHASES:
        :param generator_train_phase(str): name for generator training phase
        :param discriminator_train_phase(str): name for discriminator
            training phase

        MODEL KEYS:
        :param generator_model_key: name for generator model, e.g. "generator"
        :param discriminator_model_key: name for discriminator model,
            e.g. "discriminator"
        """
        input_batch_keys = input_batch_keys or [data_input_key]
        registered_phases = (
            (generator_train_phase, "_generator_train_phase"),
            (discriminator_train_phase, "_discriminator_train_phase"),
            (None, "_discriminator_train_phase")
        )
        super().__init__(model, device, input_batch_keys, registered_phases)

        # input keys
        self.data_input_key = data_input_key
        self.class_input_key = class_input_key
        self.noise_input_key = noise_input_key
        # output keys
        self.fake_logits_output_key = fake_logits_output_key
        self.real_logits_output_key = real_logits_output_key
        self.fake_data_output_key = fake_data_output_key
        # condition keys
        self.fake_condition_keys = fake_condition_keys or []
        self.real_condition_keys = real_condition_keys or []
        # check that discriminator will have
        # same number of arguments for real/fake data
        assert (
            len(self.fake_condition_keys) == len(self.real_condition_keys)
        ), "Number of real/fake conditions should be the same"
        # Note: this generator supports only
        # EQUALLY CONDITIONED generator (G) and discriminator (D)
        # below are some thoughts why:
        #
        # 1. G is more conditioned than D.
        #
        # it would be strange if G is conditioned on something
        # and D is NOT conditioned on same variable
        # which will most probably lead to interpreting that variable
        # as additional noise
        #
        # 2. D is more conditioned than G.
        #
        # imagine D to have additional condition 'cond_var' which is not
        # condition of G. now you have:
        #   fake_data = G(z, *other_conditions)
        #   fake_score = D(fake_data, cond_var, *other_conditions)
        # in the above example fake_data and cond_var are ~independent?
        # if they are not independent (e.g. cond_var represents
        # class condition which is fixed to the single "cat" class,
        # which may be used for finetuning pretrained GAN for specific
        # class) such configuration may have some sense
        # so they case #2 may have some sense, however for simplicity
        # it is not implemented in this Runner

        # model keys
        self.generator_key = generator_model_key
        self.discriminator_key = discriminator_model_key

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage)
        self.generator = self.model[self.generator_key]
        self.discriminator = self.model[self.discriminator_key]

    # Common utility functions

    def _get_noise_and_conditions(self):
        """Returns generator inputs"""
        z = self.state.input[self.noise_input_key]
        conditions = [
            self.state.input[key] for key in self.fake_condition_keys
        ]
        return z, conditions

    def _get_real_data_conditions(self):
        """Returns discriminator conditions (for real data)"""
        return [self.state.input[key] for key in self.real_condition_keys]

    def _get_fake_data_conditions(self):
        """Returns discriminator conditions (for fake data)"""
        return [self.state.input[key] for key in self.fake_condition_keys]

    # concrete phase methods

    def _generator_train_phase(self):
        """Forward call on generator training phase"""
        z, g_conditions = self._get_noise_and_conditions()
        d_fake_conditions = self._get_fake_data_conditions()

        fake_data = self.generator(z, *g_conditions)
        fake_logits = self.discriminator(fake_data, *d_fake_conditions)
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits
        }

    def _discriminator_train_phase(self):
        """Forward call on discriminator training phase"""
        z, g_conditions = self._get_noise_and_conditions()
        d_fake_conditions = self._get_fake_data_conditions()
        d_real_conditions = self._get_real_data_conditions()

        fake_data = self.generator(z, *g_conditions)
        fake_logits = self.discriminator(
            fake_data.detach(), *d_fake_conditions
        )
        real_logits = self.discriminator(
            self.state.input[self.data_input_key], *d_real_conditions
        )
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits,
            self.real_logits_output_key: real_logits
        }


__all__ = ["MultiPhaseRunner", "GanRunner"]
