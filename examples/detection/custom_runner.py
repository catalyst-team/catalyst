from catalyst.runners import ConfigRunner


class SSDDetectionRunner(ConfigRunner):
    """Runner for SSD models."""

    def handle_batch(self, batch):
        """Do a forward pass and compute loss.

        Args:
            batch (Dict[str, Any]): batch of data.
        """
        locs, confs = self.model(batch["image"])

        regression_loss, classification_loss = self.criterion(
            locs, batch["bboxes"], confs, batch["labels"].long()
        )
        self.batch["predicted_bboxes"] = locs
        self.batch["predicted_scores"] = confs
        self.batch_metrics["loss"] = regression_loss + classification_loss


class CenterNetDetectionRunner(ConfigRunner):
    """Runner for CenterNet models."""

    def handle_batch(self, batch):
        """Do a forward pass and compute loss.

        Args:
            batch (Dict[str, Any]): batch of data.
        """
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
        """Insert into loaders collate_fn.

        Args:
            stage (str): sage name

        Returns:
            ordered dict with torch.utils.data.DataLoader
        """
        loaders = super().get_loaders(stage)
        for item in loaders.values():
            if hasattr(item.dataset, "collate_fn"):
                item.collate_fn = item.dataset.collate_fn
        return loaders
