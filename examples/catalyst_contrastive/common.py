from catalyst import dl


class ContrastiveRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        emb1 = self.model(batch["aug1"])
        emb2 = self.model(batch["aug1"])
        self.batch = {"proj1": emb1, "proj2": emb2}
