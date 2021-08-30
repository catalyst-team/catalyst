from catalyst.runners import SupervisedRunner


class ContrastiveRunner(SupervisedRunner):
    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        sample_aug1, sample_aug2, target = batch
        projection1 = self.model(sample_aug1)
        projection2 = self.model(sample_aug2)
        self.batch = {"projection1": projection1, "projection2": projection2}
