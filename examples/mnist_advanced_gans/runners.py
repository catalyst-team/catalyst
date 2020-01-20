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
            input_batch_keys = ["features", "targets"] and your DataLoader returns 2 tensors (images and targets)
            when state.input will be {"features": batch[0], "targets": batch[1]}
        :param registered_phases: Tuple of pairs (phase_name, phase_forward_function)
            phase_forward_function's may be also str, in that case Runner should have method with same name,
            which will be called
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
        self._memorize_models(stage=stage)

    def _memorize_models(self, stage: str):
        """Maybe useful to memorize something for convenience
        e.g. self.generator = self.models["generator"]
        """
        pass


class GANRunner(MultiPhaseRunner):
    """
    Runner with logic for single-generator single-discriminator GAN training
    Various conditioning types, penalties and regularizations (such as WGAN-GP) can be easily derived from this class
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
            condition_keys: Optional[Union[str, List[str]]] = None,
            # phases:
            generator_train_phase: str = "generator_train",
            discriminator_train_phase: str = "discriminator_train",
            # model keys:
            generator_model_key: str = "generator",
            discriminator_model_key: str = "discriminator"
    ):
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
        # make self.condition_keys to be a list
        condition_keys = condition_keys or []
        self.condition_keys = [condition_keys] if isinstance(condition_keys,
                                                             str) else condition_keys

        # model keys
        self.generator_key = generator_model_key
        self.discriminator_key = discriminator_model_key

    def _memorize_models(self, stage: str):
        self.generator = self.model[self.generator_key]
        self.discriminator = self.model[self.discriminator_key]

    # common utility functions

    def _get_noise_and_conditions(self):
        z = self.state.input[self.noise_input_key]
        conditions = [self.state.input[key] for key in
                      self.condition_keys]  # TODO: maybe as dict?
        return z, conditions

    # concrete phase methods

    def _generator_train_phase(self):
        z, conditions = self._get_noise_and_conditions()

        fake_data = self.generator(z, *conditions)
        fake_logits = self.discriminator(fake_data, *conditions)
        return {
            self.fake_data_output_key: fake_data,
            self.fake_logits_output_key: fake_logits
        }

    def _discriminator_train_phase(self):
        z, conditions = self._get_noise_and_conditions()

        fake_data = self.generator(z, *conditions)
        fake_logits = self.discriminator(fake_data.detach(), *conditions)
        real_logits = self.discriminator(self.state.input[self.data_input_key],
                                         *conditions)
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
        (just add gradient penalty loss in config)
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
                 generator_train_phase: str = "generator_train",
                 discriminator_train_phase: str = "discriminator_train",
                 generator_model_key: str = "generator",
                 discriminator_model_key: str = "discriminator"):
        super().__init__(model, device, input_batch_keys, data_input_key,
                         class_input_key, noise_input_key,
                         fake_logits_output_key, real_logits_output_key,
                         fake_data_output_key, condition_keys,
                         generator_train_phase, discriminator_train_phase,
                         generator_model_key, discriminator_model_key)


BaseGANRunner = GANRunner

# class BaseGANRunner(Runner):
#     """
#     TODO: general easily extendable interface;
#     TODO 2: more general name to this class (generative models, not just GANs)"""
#     def __init__(
#         self,
#         model=None,
#         device=None,
#         features_key="features",
#         generator_key="generator",
#         discriminator_key="discriminator"
#     ):
#         super().__init__(model, device)
#
#         self.features_key = features_key
#         self.generator_key = generator_key
#         self.discriminator_key = discriminator_key
#
#     def _batch2device(self, batch: Mapping[str, Any], device):
#         if isinstance(batch, (list, tuple)):
#             batch = {self.features_key: batch[0]}
#         return super()._batch2device(batch, device)
#
#     def _prepare_for_stage(self, stage: str):
#         super()._prepare_for_stage(stage)
#         self.generator = self.model[self.generator_key]
#         self.discriminator = self.model[self.discriminator_key]
#
#         for key in [
#             "noise_dim",
#             "discriminator_train_phase",
#             "generator_train_phase"
#         ]:
#             assert hasattr(self.state, key) \
#                 and getattr(self.state, key) is not None
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
#             self.state.phase is None
#             or self.state.phase == self.state.discriminator_train_phase
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


# class GanRunner(BaseGANRunner):
#     """TODO:
#         vanilla noise2image GAN
#         z (noise) -> generator_model -> g_image
#         image -> discriminator_model -> logit confidence (of g_image vs d_image)
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
#             self.state.phase is None
#             or self.state.phase == self.state.discriminator_train_phase
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


# class WGANRunner(BaseGANRunner):
#     """TODO:
#         Wasserstein noise2image GAN
#         z (noise) -> generator_model -> g_image
#         image -> discriminator_model -> logit confidence (of g_image vs d_image)
#     """
#
#     def forward(self, batch):
#         real_features = batch[self.features_key]
#         batch_size = real_features.shape[0]
#
#         # real_targets = torch.ones((batch_size, 1), device=self.device)
#         # fake_targets = torch.zeros((batch_size, 1), device=self.device)
#         # self.state.input["real_targets"] = real_targets
#         # self.state.input["fake_targets"] = fake_targets
#         z = torch.randn((batch_size, self.state.noise_dim), device=self.device)
#
#         if (
#                 self.state.phase is None
#                 or self.state.phase == self.state.discriminator_train_phase
#         ):
#             # (None for validation mode)
#             fake_features = self.generator(z)
#             fake_validity = self.discriminator(fake_features.detach())
#             real_validity = self.discriminator(real_features)
#             # d_loss = fake_validity + real_validity (+ GP or something else if enabled)
#             # WARN: DO NOT FORGET TO IMPOSE LIPSITZ'S CONSTRAINED SOMEHOW (clip weights in optimizer/add gradient penalty, etc)
#             # TODO: how to check the condition described above? there should be some asserts
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_validity": fake_validity,
#                 "real_validity": real_validity
#             }
#         elif self.state.phase == self.state.generator_train_phase:
#             fake_features = self.generator(z)
#             fake_validity = self.discriminator(fake_features)
#             # --> g_loss (fake logits + REAL targets)
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_validity": fake_validity
#             }
#         else:
#             raise NotImplementedError(f"Unknown phase: self.state.phase")


# class WGAN_GP_Runner(BaseGANRunner):
#     """TODO:
#         Wasserstein noise2image GAN
#         z (noise) -> generator_model -> g_image
#         image -> discriminator_model -> logit confidence (of g_image vs d_image)
#     """
#
#     # TODO: implement as callback
#     @staticmethod
#     def compute_gradient_penalty(
#             discriminator,
#             fake_imgs, real_imgs,
#             device):
#         """Calculates the gradient penalty loss for WGAN GP"""
#         # Random weight term for interpolation between real and fake samples
#         alpha = torch.rand((real_imgs.size(0), 1, 1, 1), device=device)
#         # Get random interpolation between real and fake samples
#         interpolates = (alpha * real_imgs + (
#                     (1 - alpha) * fake_imgs)).requires_grad_(True)
#         interpolates.requires_grad_(True)
#         d_interpolates = discriminator(interpolates)
#         if not d_interpolates.requires_grad:
#             # TODO: deal with it (outputs does not require grad in validation mode)
#             # raise ValueError("Why the hell??? one of D inputs has requires_grad=True,"
#             #                  "so output should also have requires_grad=False")
#             return torch.zeros((real_imgs.size(0), 1), device=device)
#         fake = torch.ones((real_imgs.size(0), 1), device=device,
#                           requires_grad=False)
#         # Get gradient w.r.t. interpolates
#         gradients = torch.autograd.grad(
#             outputs=d_interpolates,
#             inputs=interpolates,
#             grad_outputs=fake,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True,
#         )[0]
#         gradients = gradients.view(gradients.size(0), -1)
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)  # .mean()
#         return gradient_penalty
#
#     def forward(self, batch):
#         real_features = batch[self.features_key]
#         batch_size = real_features.shape[0]
#
#         # real_targets = torch.ones((batch_size, 1), device=self.device)
#         # fake_targets = torch.zeros((batch_size, 1), device=self.device)
#         # self.state.input["real_targets"] = real_targets
#         # self.state.input["fake_targets"] = fake_targets
#         z = torch.randn((batch_size, self.state.noise_dim), device=self.device)
#
#         if (
#                 self.state.phase is None
#                 or self.state.phase == self.state.discriminator_train_phase
#         ):
#             # (None for validation mode)
#             fake_features = self.generator(z)
#             fake_validity = self.discriminator(fake_features.detach())
#             real_validity = self.discriminator(real_features)
#             # gradient_penalty = self.compute_gradient_penalty(
#             #     discriminator=self.discriminator,
#             #     fake_imgs=fake_features,
#             #     real_imgs=real_features,
#             #     device=self.device
#             # )
#             # d_loss = fake_validity + real_validity (+ GP or something else if enabled)
#             # TODO: how to check the condition described above? there should be some asserts
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_validity": fake_validity,
#                 "real_validity": real_validity,
#                 # "gradient_penalty": gradient_penalty
#             }
#         elif self.state.phase == self.state.generator_train_phase:
#             fake_features = self.generator(z)
#             fake_validity = self.discriminator(fake_features)
#             # --> g_loss (fake logits + REAL targets)
#             return {
#                 "fake_features": fake_features,  # visualization purposes only
#                 "fake_validity": fake_validity
#             }
#         else:
#             raise NotImplementedError(f"Unknown phase: self.state.phase")


class CGanRunner(BaseGANRunner):
    """TODO:
        vanilla noise & [one_hot_class_id_condition] to image GAN
        z (noise) & one_hot_class_id_condition -> generator_model -> g_image
        image & one_hot_class_id_condition -> discriminator_model -> logit confidence (of g_image vs d_image)
    """

    def __init__(self, model=None, device=None,
                 features_key="features",
                 targets_key="targets",
                 generator_key="generator",
                 discriminator_key="discriminator"):
        super().__init__(model, device, features_key, generator_key,
                         discriminator_key)
        self.targets_key = targets_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):  # TODO: merge into base gan runner
            batch = {self.features_key: batch[0], self.targets_key: batch[1]}
        return super()._batch2device(batch, device)

    def forward(self, batch):
        real_features = batch[self.features_key]
        real_c_inds = batch[self.targets_key]
        batch_size = real_features.shape[0]

        n_classes = 10
        real_c = torch.zeros((batch_size, n_classes), device=self.device)
        real_c[torch.arange(batch_size, device=self.device), real_c_inds] = 1

        real_targets = torch.ones((batch_size, 1), device=self.device)
        fake_targets = torch.zeros((batch_size, 1), device=self.device)
        self.state.input["real_targets"] = real_targets
        self.state.input["fake_targets"] = fake_targets
        z = torch.randn((batch_size, self.state.noise_dim), device=self.device)

        # TODO: refactor as callback with on_batch_start
        n_classes = 10
        c = torch.zeros((batch_size, n_classes))
        c_one_hot_indices = torch.randint(0, 10, (batch_size,))
        c[torch.arange(batch_size), c_one_hot_indices] = 1
        fake_c = c.to(self.device)

        if (
                self.state.phase is None
                or self.state.phase == self.state.discriminator_train_phase
        ):
            # (None for validation mode)
            fake_features = self.generator(z, fake_c)
            fake_logits = self.discriminator(fake_features.detach(), fake_c)
            real_logits = self.discriminator(real_features, real_c)
            # --> d_loss
            # (fake logits + FAKE targets) + (real logits + real targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits,
                "real_logits": real_logits
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z, fake_c)
            fake_logits = self.discriminator(fake_features, fake_c)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")


class ICGanRunner(CGanRunner):
    """TODO:
        vanilla noise & [other_image_of_that_class] to image GAN
        z (noise) -> generator_model -> g_image
        image -> discriminator_model -> logit confidence (of g_image vs d_image)
    """

    def forward(self, batch):
        real_features = batch[self.features_key]
        real_targets = batch[self.targets_key]
        batch_size = real_features.shape[0]
        assert torch.equal(real_targets[:batch_size // 2],
                           real_targets[batch_size // 2:])
        same_class_real_features = torch.cat(
            (real_features[batch_size // 2:], real_features[:batch_size // 2]),
            dim=0)
        self.state.input["same_real_images"] = same_class_real_features
        z = torch.randn((batch_size, self.state.noise_dim), device=self.device)

        if (
                self.state.phase is None
                or self.state.phase == self.state.discriminator_train_phase
        ):
            # (None for validation mode)
            fake_features = self.generator(z, same_class_real_features)
            fake_logits = self.discriminator(fake_features.detach(),
                                             same_class_real_features)
            real_logits = self.discriminator(real_features,
                                             same_class_real_features)
            # --> d_loss
            # (fake logits + FAKE targets) + (real logits + real targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits,
                "real_logits": real_logits
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z, same_class_real_features)
            fake_logits = self.discriminator(fake_features,
                                             same_class_real_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")


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


class D2GanRunner(BaseGANRunner):
    """TODO:
        vanilla noise  to image GAN *with two discriminators*
        z (noise) -> generator_model -> g_image
        D_adversarial: image -> D_adversarial -> logit confidence (of g_image vs d_image)
        D_classification: image -> D_classifier -> classification_score
    """

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
                "real_logits": real_logits
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z)
            fake_logits = self.discriminator(fake_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")
