from typing import Mapping, Any

import torch
from catalyst.dl import Runner


class GanRunner(Runner):
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


__all__ = ["GanRunner"]
