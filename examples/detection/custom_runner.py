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
