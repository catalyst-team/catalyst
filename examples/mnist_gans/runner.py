from typing import Mapping, Any

import torch
from catalyst.dl import Runner


class GANRunner(Runner):
    def __init__(
        self,
        model=None,
        device=None,
        images_key="images",
        targets_key="targets",
        model_generator_key="generator",
        model_discriminator_key="discriminator"
    ):
        super().__init__(model, device)

        self.images_key = images_key
        self.targets_key = targets_key

        self.model_generator_key = model_generator_key
        self.model_discriminator_key = model_discriminator_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            assert len(batch) == 2
            batch = {self.images_key: batch[0], self.targets_key: batch[1]}
        return super()._batch2device(batch, device)

    def _run_prestage(self, stage: str):
        self.generator = self.model[self.model_generator_key]
        self.discriminator = self.model[self.model_discriminator_key]

    def predict_batch(self, batch):
        real_imgs, _ = batch["images"], batch["targets"]

        real_targets = torch.ones((real_imgs.size(0), 1), device=self.device)
        fake_targets = 1 - real_targets
        self.state.input["fake_targets"] = fake_targets
        self.state.input["real_targets"] = real_targets
        z = torch.randn(
            (real_imgs.size(0), self.generator.noise_dim), device=self.device
        )

        if (
            self.state.phase is None
            or self.state.phase == "discriminator_train"
        ):
            # (None for validation mode)
            fake_imgs = self.generator(z)
            fake_logits = self.discriminator(fake_imgs.detach())
            real_logits = self.discriminator(real_imgs)
            # --> d_loss
            # (fake logits + FAKE targets) + (real logits + real targets)
            return {
                "fake_images": fake_imgs,  # visualization purposes only
                "fake_logits": fake_logits,
                "real_logits": real_logits
            }
        else:
            fake_imgs = self.generator(z)
            fake_logits = self.discriminator(fake_imgs)
            # --> g_loss (fake logits + REAL targets)
            return {
                "fake_images": fake_imgs,  # visualization purposes only
                "fake_logits": fake_logits
            }
