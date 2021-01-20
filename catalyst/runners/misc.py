from catalyst.core.runner import IStageBasedRunner


class SupervisedRunnerV2(IStageBasedRunner):
    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits.view(-1),
        }
