from typing import Mapping, Any

import torch
from catalyst.dl import Runner


class MultiStageRunner(Runner):
    def __init__(self, model=None, device=None):
        super().__init__(model, device)
        self._callbacks = None

    @property
    def callbacks(self):
        if self.phase_manager is None:
            return self._callbacks
        else:
            return self.phase_manager.get_callbacks(self.state)

    @callbacks.setter
    def callbacks(self, callbacks):
        self._callbacks = callbacks

    def _run_prestage(self, stage: str):
        if hasattr(self.experiment, "get_phase_manager"):
            self.phase_manager = self.experiment.get_phase_manager(stage)
        else:
            self.phase_manager = None

    def _run_stage(self, stage: str):
        # @TODO rewrite parent (Runner) method instead
        self._prepare_state(stage)
        loaders = self.experiment.get_loaders(stage)
        self.callbacks = self.experiment.get_callbacks(stage)

        self._run_prestage(stage)

        self._run_event("stage_start")
        for epoch in range(self.state.num_epochs):
            self.state.stage_epoch = epoch

            self._run_event("epoch_start")
            self._run_epoch(loaders)
            self._run_event("epoch_end")

            if self._check_run and self.state.epoch >= 3:
                break
            if self.state.early_stop:
                self.state.early_stop = False
                break

            self.state.epoch += 1
        self._run_event("stage_end")

    def _run_batch(self, batch):
        self.state.phase = self.phase_manager.get_phase_name(self.state)
        super()._run_batch(batch)
        self.phase_manager.step(self.state)


class GANRunner(MultiStageRunner):
    def __init__(self, model=None, device=None,
                 images_key="images",
                 targets_key="targets"):
        super().__init__(model, device)
        self.images_key = images_key
        self.targets_key = targets_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            assert len(batch) == 2
            batch = {self.images_key: batch[0], self.targets_key: batch[1]}
        return super()._batch2device(batch, device)

    def _run_prestage(self, stage: str):
        super()._run_prestage(stage)
        self.generator = self.model["generator"]
        self.discriminator = self.model["discriminator"]

    def predict_batch(self, batch):
        real_imgs, _ = batch["images"], batch["targets"]

        real_targets = torch.ones((real_imgs.size(0), 1), device=self.device)
        fake_targets = 1 - real_targets
        self.state.input["fake_targets"] = fake_targets
        self.state.input["real_targets"] = real_targets
        z = torch.randn((real_imgs.size(0), self.generator.noise_dim),
                        device=self.device)

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
