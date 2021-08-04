from catalyst.runners import ConfigRunner


class SSDDetectionRunner(ConfigRunner):
    def handle_batch(self, batch):
        locs, confs = self.model(batch["image"])

        regression_loss, classification_loss = self.criterion(
            locs, batch["bboxes"], confs, batch["labels"].long()
        )
        self.batch["predicted_bboxes"] = locs
        self.batch["predicted_scores"] = confs
        self.batch_metrics["loss"] = regression_loss + classification_loss


class CenterNetDetectionRunner(ConfigRunner):
    def handle_batch(self, batch):

        heatmaps, regression = self.model(batch["image"])

        loss, mask_loss, regression_loss = self.criterion(
            heatmaps, regression, batch["heatmap"], batch["wh_regr"]
        )

        self.batch["predicted_heatmap"] = heatmaps
        self.batch["predicted_regression"] = regression

        self.batch_metrics["mask_loss"] = mask_loss.item()
        self.batch_metrics["regression_loss"] = regression_loss.item()
        self.batch_metrics["loss"] = loss

    def get_loaders(self, stage: str):
        loaders = super().get_loaders(stage)
        for item in loaders.values():
            if hasattr(item.dataset, "collate_fn"):
                item.collate_fn = item.dataset.collate_fn
        return loaders
