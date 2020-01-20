from typing import Any, Mapping, Optional, Union, List, Tuple, Callable, \
    Dict  # isort:skip

import torch

from catalyst.dl import Runner, SupervisedRunner
from utils.typing import Model, Device


class MultiPhaseRunner(Runner):
    """
    Base Runner with multiple phases
    """

    def __init__(
            self,
            model: Union[Model, Dict[str, Model]] = None,
            device: Device = None,
            input_batch_keys: Optional[List[str]] = None,
            registered_phases: Tuple[
                Tuple[str, Union[str, Callable]], ...] = None
    ):
        """

        :param model:
        :param device:
        :param input_batch_keys: list of strings of keys for batch elements, e.g.
            input_batch_keys = ["features", "targets"] and your
            DataLoader returns 2 tensors (images and targets)
            when state.input will be {"features": batch[0], "targets": batch[1]}
        :param registered_phases:
            Tuple of pairs (phase_name, phase_forward_function)
            phase_forward_function's may be also str, in that case Runner
            should have method with same name, which will be called
        """
        super().__init__(model, device)

        self.input_batch_keys = input_batch_keys

        self.registered_phases = dict()
        for phase_name, phase_batch_forward_fn in registered_phases:
            if not (isinstance(phase_name, str) or phase_name is None):
                raise ValueError(
                    f"phase '{phase_name}' of type '{type(phase_name)}' "
                    f"not supported, must be str of None")
            if phase_name in self.registered_phases:
                raise ValueError(f"phase '{phase_name}' already registered")
            if isinstance(phase_batch_forward_fn, str):
                assert hasattr(self, phase_batch_forward_fn)
                phase_batch_forward_fn = getattr(self, phase_batch_forward_fn)
            assert isinstance(phase_batch_forward_fn,
                              Callable), "must be callable"
            self.registered_phases[phase_name] = phase_batch_forward_fn

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            assert len(batch) >= len(self.input_batch_keys)
            batch = {key: value for key, value in
                     zip(self.input_batch_keys, batch)}
        return super()._batch2device(batch, device)

    def forward(self, batch, **kwargs):
        if self.state.phase not in self.registered_phases:
            raise ValueError(f"Unknown phase: '{self.state.phase}'")

        return self.registered_phases[self.state.phase]()

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage)
        self._alias_inner_params(stage=stage)

    def _alias_inner_params(self, stage: str):
        """Maybe useful to memorize something for convenience
        e.g. self.generator = self.models["generator"]
        """
        pass


class GANRunner(MultiPhaseRunner):
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
            condition_keys: Union[str, List[str]] = None,
            d_fake_condition_keys: List[str] = None,
            d_real_condition_keys: List[str] = None,
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
        :param input_batch_keys: list of strings of keys for batch elements, e.g.
            input_batch_keys = ["features", "targets"] and
            your DataLoader returns 2 tensors (images and targets)
            when state.input will be {"features": batch[0], "targets": batch[1]}

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
        :param condition_keys: list of all conditional inputs of generator
            (appear in same order as in generator model forward() call)
        :param d_fake_condition_keys: list of all conditional inputs of
            discriminator (fake data conditions)
            (appear in same order as in generator model forward() call)
        :param d_real_condition_keys: list of all conditional inputs of
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
        )  # TODO: я где-то налажал в typing что ли?
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
        self.__init_condition_keys(condition_keys,
                                   d_real_condition_keys,
                                   d_fake_condition_keys)

        # model keys
        self.generator_key = generator_model_key
        self.discriminator_key = discriminator_model_key

    def __init_condition_keys(self, condition_keys,
                              d_real_condition_keys,
                              d_fake_condition_keys):
        """
        Condition keys initialization
        :param condition_keys:
        :param d_real_condition_keys:
        :param d_fake_condition_keys:
        :return:
        """

        # make self.condition_keys to be a list
        condition_keys = condition_keys or []
        self.condition_keys = (
            [condition_keys] if isinstance(condition_keys, str)
            else condition_keys
        )

        assert (
                (d_fake_condition_keys is None)
                == (d_real_condition_keys is None)
        ), "'d_fake_condition_keys' and 'd_real_condition_keys' " \
           "should be either None or not None at same time"

        if d_fake_condition_keys is not None:
            # this generator supports only
            # SAME CONDITIONED generator (G) and discriminator (D)
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

            assert all(
                key in self.condition_keys for key in d_fake_condition_keys
            ), "all discriminator conditions must be generator conditions"
            assert all(
                key in d_fake_condition_keys for key in self.condition_keys
            ), "all generator conditions must be discriminator conditions"
            # additional check that discriminator will have
            # same number of arguments for real/fake data
            assert len(d_fake_condition_keys) == len(d_real_condition_keys), \
                "Not the same number of real/fake conditions"

        if not self.condition_keys:
            # simple case: no condition
            d_real_condition_keys = []
            d_fake_condition_keys = []

        self.d_real_condition_keys = d_real_condition_keys
        self.d_fake_condition_keys = d_fake_condition_keys

    def _alias_inner_params(self, stage: str):
        self.generator = self.model[self.generator_key]
        self.discriminator = self.model[self.discriminator_key]

    # common utility functions

    def _get_noise_and_conditions(self):
        """returns generator inputs"""
        z = self.state.input[self.noise_input_key]
        conditions = [self.state.input[key] for key in
                      self.condition_keys]  # TODO: maybe as dict?
        return z, conditions

    def _get_real_data_conditions(self):
        """returns discriminator conditions (for real data)"""
        return [self.state.input[key] for key in self.d_real_condition_keys]

    def _get_fake_data_conditions(self):
        """returns discriminator conditions (for fake data)"""
        return [self.state.input[key] for key in self.d_fake_condition_keys]

    # concrete phase methods

    def _generator_train_phase(self):
        """forward() on generator training phase"""
        z, g_conditions = self._get_noise_and_conditions()
        d_fake_conditions = self._get_fake_data_conditions()

        fake_data = self.generator(z, *g_conditions)
        fake_logits = self.discriminator(fake_data, *d_fake_conditions)
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits
        }

    def _discriminator_train_phase(self):
        """forward() on discriminator training phase"""
        z, g_conditions = self._get_noise_and_conditions()
        d_fake_conditions = self._get_fake_data_conditions()
        d_real_conditions = self._get_real_data_conditions()

        fake_data = self.generator(z, *g_conditions)
        fake_logits = self.discriminator(fake_data.detach(), *d_fake_conditions)
        real_logits = self.discriminator(self.state.input[self.data_input_key],
                                         *d_real_conditions)
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits,
            self.real_logits_output_key: real_logits
        }


class WGANRunner(GANRunner):
    """
    Wasserstein GAN Runner
        Note: all the changes made compared to GANRunner
            are just renaming some input/output keys to conventional ones
    Also this runner may be used unchanged for WGAN-GP
        (just add gradient penalty loss in yaml config)
    """

    def __init__(self, model: Union[Model, Dict[str, Model]] = None,
                 device: Device = None,
                 input_batch_keys: Optional[List[str]] = None,
                 data_input_key: str = "data",
                 class_input_key: str = "class_targets",
                 noise_input_key: str = "noise",
                 fake_logits_output_key: str = "fake_validity",
                 real_logits_output_key: str = "real_validity",
                 fake_data_output_key: str = "fake_data",
                 condition_keys: Optional[Union[str, List[str]]] = None,
                 d_fake_condition_keys: List[str] = None,
                 d_real_condition_keys: List[str] = None,
                 generator_train_phase: str = "generator_train",
                 discriminator_train_phase: str = "discriminator_train",
                 generator_model_key: str = "generator",
                 discriminator_model_key: str = "discriminator"):
        super().__init__(
            model=model,
            device=device,
            input_batch_keys=input_batch_keys,
            data_input_key=data_input_key,
            class_input_key=class_input_key,
            noise_input_key=noise_input_key,
            fake_logits_output_key=fake_logits_output_key,
            real_logits_output_key=real_logits_output_key,
            fake_data_output_key=fake_data_output_key,
            condition_keys=condition_keys,
            d_fake_condition_keys=d_fake_condition_keys,
            d_real_condition_keys=d_real_condition_keys,
            generator_train_phase=generator_train_phase,
            discriminator_train_phase=discriminator_train_phase,
            generator_model_key=generator_model_key,
            discriminator_model_key=discriminator_model_key)


class CGanRunner(GANRunner):
    """
    (Class) Conditional GAN
        both generator and discriminator are conditioned on one-hot class target
    """

    def __init__(self, model: Union[Model, Dict[str, Model]] = None,
                 device: Device = None,
                 data_input_key: str = "data",
                 class_input_key: str = "class_targets",
                 noise_input_key: str = "noise",
                 fake_logits_output_key: str = "fake_logits",
                 real_logits_output_key: str = "real_logits",
                 fake_data_output_key: str = "fake_data",
                 d_fake_condition_key: str = "class_targets_one_hot",
                 d_real_condition_key: str = "class_targets_one_hot",
                 generator_train_phase: str = "generator_train",
                 discriminator_train_phase: str = "discriminator_train",
                 generator_model_key: str = "generator",
                 discriminator_model_key: str = "discriminator"):
        input_batch_keys = [data_input_key, class_input_key]
        condition_keys = [d_fake_condition_key]
        d_fake_condition_keys = [d_fake_condition_key]
        d_real_condition_keys = [d_real_condition_key]
        super().__init__(
            model=model,
            device=device,
            input_batch_keys=input_batch_keys,
            data_input_key=data_input_key,
            class_input_key=class_input_key,
            noise_input_key=noise_input_key,
            fake_logits_output_key=fake_logits_output_key,
            real_logits_output_key=real_logits_output_key,
            fake_data_output_key=fake_data_output_key,
            condition_keys=condition_keys,
            d_fake_condition_keys=d_fake_condition_keys,
            d_real_condition_keys=d_real_condition_keys,
            generator_train_phase=generator_train_phase,
            discriminator_train_phase=discriminator_train_phase,
            generator_model_key=generator_model_key,
            discriminator_model_key=discriminator_model_key)


class ICGanRunner(CGanRunner):
    """
    (Image) Conditional GAN
        both generator and discriminator are conditioned on same class image
        assumed usage:
            another_image_of_class_c = generator(noise, image_of_class_c)
            real_score = discriminator(image1_class_c, image2_class_c)
            fake_score = discriminator(gen_image_class_c, image2_class_c)
    """

    def __init__(self, model: Union[Model, Dict[str, Model]] = None,
                 device: Device = None, data_input_key: str = "data",
                 class_input_key: str = "class_targets",
                 noise_input_key: str = "noise",
                 fake_logits_output_key: str = "fake_logits",
                 real_logits_output_key: str = "real_logits",
                 fake_data_output_key: str = "fake_data",
                 d_fake_condition_key: str = "same_class_data",
                 d_real_condition_key: str = "same_class_data",
                 generator_train_phase: str = "generator_train",
                 discriminator_train_phase: str = "discriminator_train",
                 generator_model_key: str = "generator",
                 discriminator_model_key: str = "discriminator"):
        super().__init__(model=model,
                         device=device,
                         data_input_key=data_input_key,
                         class_input_key=class_input_key,
                         noise_input_key=noise_input_key,
                         fake_logits_output_key=fake_logits_output_key,
                         real_logits_output_key=real_logits_output_key,
                         fake_data_output_key=fake_data_output_key,
                         d_fake_condition_key=d_fake_condition_key,
                         d_real_condition_key=d_real_condition_key,
                         generator_train_phase=generator_train_phase,
                         discriminator_train_phase=discriminator_train_phase,
                         generator_model_key=generator_model_key,
                         discriminator_model_key=discriminator_model_key)


class AE_Runner(SupervisedRunner):
    """TODO:
        image -> autoencoder -> out_image
    """

    def __init__(self, model: Model = None, device: Device = None,
                 input_key: str = "features",
                 output_key: str = "reconstructed_features",
                 input_target_key: str = "targets"):
        super().__init__(model, device, input_key, output_key, input_target_key)


class YAERunner(Runner):
    """TODO: check if it works (NOT REALLY WORKS RIGHT NOW)"""
    INPUT_IMG_KEY = "images"
    INPUT_Y_KEY = "targets_a"
    INPUT_RANDOM_Y_KEY = "targets_b"

    OUTPUT_IMAGES_A_KEY = "images_a"
    OUTPUT_IMAGES_B_KEY = "images_b"
    OUTPUT_LOGITS_A_KEY = "logits_a"
    OUTPUT_LOGITS_B_KEY = "logits_b"

    OUTPUT_IMPLICIT_LOSS_KEY = "implicit_loss"

    def __init__(
            self,
            model: Model = None,
            device: Device = None,
            input_img_key=INPUT_IMG_KEY,
            input_y_key=INPUT_Y_KEY,
            input_random_y_key=INPUT_RANDOM_Y_KEY,
            output_images_a_key=OUTPUT_IMAGES_A_KEY,
            output_images_b_key=OUTPUT_IMAGES_B_KEY,
            output_logits_a_key=OUTPUT_LOGITS_A_KEY,
            output_logits_b_key=OUTPUT_LOGITS_B_KEY,
            output_implicit_loss_key=OUTPUT_IMPLICIT_LOSS_KEY
    ):
        """
        Custom runner

        :param model: model
            with .encoder(x) -> x_explicit, x_implicit
            and  .decoder(y, x_implicit) -> x
        :param device:

        :param input_img_key:
        :param input_y_key:
        :param input_random_y_key:

        :param output_images_a_key:
        :param output_images_b_key:
        :param output_logits_a_key:
        :param output_logits_b_key:
        """
        super().__init__(model=model, device=device)
        self.input_img_key = input_img_key
        self.input_y_key = input_y_key
        self.input_random_y_key = input_random_y_key

        self.input_key = (
            self.input_img_key, self.input_y_key, self.input_random_y_key)

        self.output_images_a_key = output_images_a_key
        self.output_images_b_key = output_images_b_key
        self.output_logits_a_key = output_logits_a_key
        self.output_logits_b_key = output_logits_b_key

        self.output_implicit_loss_key = output_implicit_loss_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        batch = super()._batch2device(batch, device)
        assert len(batch) == len(self.input_key)
        return dict((k, v) for k, v in zip(self.input_key, batch))

    def predict_batch(self, batch: Mapping[str, Any]):
        images = batch[self.input_img_key]
        targets_a = batch[self.input_y_key]
        targets_b = batch[self.input_random_y_key]

        enc = self.model.encoder
        dec = self.model.decoder
        #
        expl_a, impl_a = enc(images)

        images_a = dec(targets_a, impl_a)
        expl_aa, impl_aa = enc(images_a)

        images_b = dec(targets_b, impl_a)
        expl_ab, impl_ab = enc(images_b)

        impl_loss = torch.norm(impl_aa - impl_ab, dim=1)
        if self.state.stage.startswith('sampl') or self.state.stage.startswith(
                'infer'):
            return {
                self.output_images_a_key: images_a,
                self.output_images_b_key: images_b,
                self.output_logits_a_key: expl_a,
                self.output_logits_b_key: expl_ab,
                self.output_implicit_loss_key: impl_loss,
                # inference sampling needs this
                self.input_y_key: targets_a,
                self.input_random_y_key: targets_b
            }
        return {
            self.output_images_a_key: images_a,
            self.output_images_b_key: images_b,
            self.output_logits_a_key: expl_a,
            self.output_logits_b_key: expl_ab,
            self.output_implicit_loss_key: impl_loss,
        }

# class D2GanRunner(BaseGANRunner):
#     """TODO:
#         vanilla noise  to image GAN *with two discriminators*
#         z (noise) -> generator_model -> g_image
#         D_adversarial: image -> D_adversarial -> logit confidence (of g_image vs d_image)
#         D_classification: image -> D_classifier -> classification_score
#     """
#
#     def forward(self, batch):
#         real_features = batch[self.features_key]
#         batch_size = real_features.shape[0]
#
#         real_targets = torch.ones((batch_size, 1), device=self.device)
#         fake_targets = torch.zeros((batch_size, 1), device=self.device)
#         self.state.input["real_targets"] = real_targets
#         self.state.input["fake_targets"] = fake_targets
#         z = torch.randn((batch_size, self.state.noise_dim), device=self.device)
#
#         if (
#                 self.state.phase is None
#                 or self.state.phase == self.state.discriminator_train_phase
#         ):
#             # (None for validation mode)
#             fake_features = self.generator(z)
#             fake_logits = self.discriminator(fake_features.detach())
#             real_logits = self.discriminator(real_features)
#             # --> d_loss
#             # (fake logits + FAKE targets) + (real logits + real targets)
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_logits": fake_logits,
#                 "real_logits": real_logits
#             }
#         elif self.state.phase == self.state.generator_train_phase:
#             fake_features = self.generator(z)
#             fake_logits = self.discriminator(fake_features)
#             # --> g_loss (fake logits + REAL targets)
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_logits": fake_logits
#             }
#         else:
#             raise NotImplementedError(f"Unknown phase: self.state.phase")
