from typing import Any, Mapping  # isort:skip

import torch

from catalyst.dl import Runner


class BaseGANRunner(Runner):
    """
    TODO: general easily extendable interface;
    TODO 2: more general name to this class (generative models, not just GANs)"""
    def __init__(
        self,
        model=None,
        device=None,
        features_key="features",
        generator_key="generator",
        discriminator_key="discriminator"
    ):
        super().__init__(model, device)

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
            "generator_train_phase"
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


class GanRunner(BaseGANRunner):
    """TODO:
        vanilla noise2image GAN
        z (noise) -> generator_model -> g_image
        image -> discriminator_model -> logit confidence (of g_image vs d_image)
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


class WGANRunner(BaseGANRunner):
    """TODO:
        Wasserstein noise2image GAN
        z (noise) -> generator_model -> g_image
        image -> discriminator_model -> logit confidence (of g_image vs d_image)
    """

    def forward(self, batch):
        real_features = batch[self.features_key]
        batch_size = real_features.shape[0]

        # real_targets = torch.ones((batch_size, 1), device=self.device)
        # fake_targets = torch.zeros((batch_size, 1), device=self.device)
        # self.state.input["real_targets"] = real_targets
        # self.state.input["fake_targets"] = fake_targets
        z = torch.randn((batch_size, self.state.noise_dim), device=self.device)

        if (
                self.state.phase is None
                or self.state.phase == self.state.discriminator_train_phase
        ):
            # (None for validation mode)
            fake_features = self.generator(z)
            fake_validity = self.discriminator(fake_features.detach())
            real_validity = self.discriminator(real_features)
            # d_loss = fake_validity + real_validity (+ GP or something else if enabled)
            # WARN: DO NOT FORGET TO IMPOSE LIPSITZ'S CONSTRAINED SOMEHOW (clip weights in optimizer/add gradient penalty, etc)
            # TODO: how to check the condition described above? there should be some asserts
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_validity": fake_validity,
                "real_validity": real_validity
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z)
            fake_validity = self.discriminator(fake_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_validity": fake_validity
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")


class WGAN_GP_Runner(BaseGANRunner):
    """TODO:
        Wasserstein noise2image GAN
        z (noise) -> generator_model -> g_image
        image -> discriminator_model -> logit confidence (of g_image vs d_image)
    """

    # TODO: implement as callback
    @staticmethod
    def compute_gradient_penalty(
            discriminator,
            fake_imgs, real_imgs,
            device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_imgs.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        interpolates.requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        if not d_interpolates.requires_grad:
            # TODO: deal with it (outputs does not require grad in validation mode)
            # raise ValueError("Why the hell??? one of D inputs has requires_grad=True,"
            #                  "so output should also have requires_grad=False")
            return torch.zeros((real_imgs.size(0), 1), device=device)
        fake = torch.ones((real_imgs.size(0), 1), device=device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)#.mean()
        return gradient_penalty

    def forward(self, batch):
        real_features = batch[self.features_key]
        batch_size = real_features.shape[0]

        # real_targets = torch.ones((batch_size, 1), device=self.device)
        # fake_targets = torch.zeros((batch_size, 1), device=self.device)
        # self.state.input["real_targets"] = real_targets
        # self.state.input["fake_targets"] = fake_targets
        z = torch.randn((batch_size, self.state.noise_dim), device=self.device)

        if (
                self.state.phase is None
                or self.state.phase == self.state.discriminator_train_phase
        ):
            # (None for validation mode)
            fake_features = self.generator(z)
            fake_validity = self.discriminator(fake_features.detach())
            real_validity = self.discriminator(real_features)
            # gradient_penalty = self.compute_gradient_penalty(
            #     discriminator=self.discriminator,
            #     fake_imgs=fake_features,
            #     real_imgs=real_features,
            #     device=self.device
            # )
            # d_loss = fake_validity + real_validity (+ GP or something else if enabled)
            # TODO: how to check the condition described above? there should be some asserts
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_validity": fake_validity,
                "real_validity": real_validity,
                # "gradient_penalty": gradient_penalty
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z)
            fake_validity = self.discriminator(fake_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_validity": fake_validity
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")


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
        super().__init__(model, device, features_key, generator_key, discriminator_key)
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
        assert torch.equal(real_targets[:batch_size//2], real_targets[batch_size//2:])
        same_class_real_features = torch.cat((real_features[batch_size//2:], real_features[:batch_size//2]), dim=0)

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
            fake_features = self.generator(z, same_class_real_features)
            fake_logits = self.discriminator(fake_features.detach(), same_class_real_features)
            real_logits = self.discriminator(real_features, same_class_real_features)
            # --> d_loss
            # (fake logits + FAKE targets) + (real logits + real targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits,
                "real_logits": real_logits
            }
        elif self.state.phase == self.state.generator_train_phase:
            fake_features = self.generator(z, same_class_real_features)
            fake_logits = self.discriminator(fake_features, same_class_real_features)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_features": fake_features,  # visualization purposes only
                "fake_logits": fake_logits
            }
        else:
            raise NotImplementedError(f"Unknown phase: self.state.phase")


class AE_Runner(BaseGANRunner):
    """TODO:
        image + [additional_inputs] -> autoencoder -> out_image
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


class YAE_Runner(BaseGANRunner):
    """TODO:
        e=encoder; d=decoder
        image + [additional_inputs] -> autoencoder -> out_image + [additional_outputs]
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
