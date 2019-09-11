from typing import Tuple, Mapping, Any

import torch
from catalyst.dl import RunnerState, Runner
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class GANRunner(Runner):

    def __init__(self, model=None, device=None, images_key="images", targets_key="targets"):
        super().__init__(model, device)
        self.images_key = images_key
        self.targets_key = targets_key

    # def _get_experiment_components(
    #         self,
    #         stage: str = None
    # ) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
    #     result = super()._get_experiment_components(stage)
    #
    #     return result

    @property
    def callbacks(self):
        return self._get_callbacks()

    @callbacks.setter
    def callbacks(self, value):
        pass

    def _run_stage(self, stage: str):
        self._prepare_state(stage)
        loaders = self.experiment.get_loaders(stage)
        # self.callbacks = self.experiment.get_callbacks(stage)
        self.generator = self.model["generator"]
        self.discriminator = self.model["discriminator"]
        # @TODO ungovnocode
        self.phase_manager = self.experiment.get_phase_manager(stage)
        #

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

    def _get_callbacks(self):
        return self.phase_manager.get_callbacks(self.state)

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (list, tuple)):
            assert len(batch) == 2
            batch = {self.images_key: batch[0], self.targets_key: batch[1]}
        return super()._batch2device(batch, device)

    def _run_batch(self, batch):
        self.state.phase = self.phase_manager.get_phase_name(self.state)
        super()._run_batch(batch)
        self.phase_manager.step(self.state)

    def predict_batch(self, batch):
        real_imgs, labels = batch["images"], batch["targets"]

        real_targets = torch.ones((real_imgs.size(0), 1), device=self.device)
        fake_targets = 1 - real_targets
        self.state.input["fake_targets"] = fake_targets
        self.state.input["real_targets"] = real_targets
        z = torch.randn((real_imgs.size(0), self.generator.noise_dim), device=self.device)

        if self.state.phase is None or self.state.phase == "discriminator_train":
            # (None for validation mode)
            fake_imgs = self.generator(z)
            fake_logits = self.discriminator(fake_imgs.detach())
            real_logits = self.discriminator(real_imgs)
            # --> d_loss (fake logits + FAKE targets) + (real logits + real targets)
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
