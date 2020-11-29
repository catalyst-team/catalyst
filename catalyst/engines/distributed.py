from device import DeviceEngine

import torch.nn as nn


class DistributedDeviceEngine(DeviceEngine):
    # TODO: maybe handle unpacking DataParallel to simple nn.Module

    def load_checkpoint(
        self,
        file: str,
        model: nn.DataParallel,
        optimizer: nn.Module = None,
        criterion=None,
        scheduler=None,
    ):
        content = torch.load(file)

        if "model_state_dict" in content:
            model.module.load_state_dict(content["model_state_dict"])

        if "optimizer_state_dict" in content and optimizer is not None:
            optimizer.load_state_dict(content["optimizer_state_dict"])

        if "criterion_state_dict" in content and criterion is not None:
            criterion.load_state_dict(content["criterion_state_dict"])

        if "scheduler_state_dict" in content and scheduler is not None:
            scheduler.load_state_dict(content["scheduler_state_dict"])
