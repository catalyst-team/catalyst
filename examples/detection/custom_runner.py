from collections import OrderedDict

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from catalyst.runners import ConfigRunner

from .dataset import DetectionDataset


class SSDDetectionRunner(ConfigRunner):
    def get_datasets(self, stage: str):
        train_json = "/mnt/4tb/datasets/star/b405_v2/train/dataset.json"
        train_imgs_dir = "/mnt/4tb/datasets/star/b405_v2/train/images"
        valid_json = "/mnt/4tb/datasets/star/b405_v2/test/dataset.json"
        valid_imgs_dir = "/mnt/4tb/datasets/star/b405_v2/test/images"

        datasets = OrderedDict()
        datasets["train"] = DetectionDataset(
            train_json,
            train_imgs_dir,
            transforms=albu.Compose(
                [albu.Resize(300, 300), albu.Normalize(), albu.HorizontalFlip(p=0.5), ToTensorV2()]
            ),
        )
        datasets["valid"] = DetectionDataset(
            valid_json,
            valid_imgs_dir,
            transforms=albu.Compose([albu.Resize(300, 300), albu.Normalize(), ToTensorV2()]),
        )
        return datasets

    def handle_batch(self, batch):
        locs, confs = self.model(batch["image"])
        # print(">>>", locs.shape, locs.dtype, batch["bboxes"].shape, batch["bboxes"].dtype)
        # print(">>>", confs.shape, confs.dtype, batch["labels"].shape, batch["labels"].dtype)

        regression_loss, classification_loss = self.criterion(
            locs, batch["bboxes"], confs, batch["labels"].long()
        )
        self.batch["predicted_bboxes"] = locs
        self.batch["predicted_scores"] = confs
        self.batch_metrics["loss"] = regression_loss + classification_loss
