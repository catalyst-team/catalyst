from typing import Any, Dict, Mapping, Union

import apex.amp as amp
import torch

from catalyst.engines.device import DeviceEngine


class APEXEngine(DeviceEngine):
    # TODO: make clickable link about opt_level's
    def __init__(self, device: str = "cuda", opt_level: str = "O1"):
        """
        Args:
            device (str): use device, default is `"cpu"`.
            opt_level (str): optimization level, should be one of
                "O0", "O1", "O2", "O3" or "O4".

                    - "O0" - no-op training
                    - "O1" - mixed precision (FP16) training
                    - "O2" - "almost" mixed precision training
                    - "O3" - another implementation of mixed precision training

                Details about levels can be found here:
                    https://nvidia.github.io/apex/amp.html#opt-levels

                Default is "O1".
        """
        super().__init__(device)
        self.opt_level = opt_level

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}',opt_level='{self.opt_level}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        # TODO: how could we do better?)
        # model
        model = model_fn()
        model = self.sync_device(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn(model=model)
        optimizer = self.sync_device(optimizer)

        # from official docs:
        #   https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        model, optimizer = amp.initialize(model, optimizer, opt_level=self.opt_level)

        # scheduler
        scheduler = scheduler_fn(optimizer=optimizer)
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def backward_loss(self, loss, model, optimizer) -> None:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            # NOTE: propper way to save state, docs:
            #   https://nvidia.github.io/apex/amp.html#checkpointing
            "amp": amp.state_dict(),
            **kwargs,
        }

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "criterion_state_dict" in checkpoint and criterion is not None:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])

        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # NOTE: propper way to load state, docs:
        #   https://nvidia.github.io/apex/amp.html#checkpointing
        if "amp" in checkpoint:
            amp.load_state_dict(checkpoint["amp"])
