from catalyst.dl import Runner
from catalyst.dl.utils import is_wrapped_with_ddp


class HuggingfaceRunner(Runner):
    """
    @TODO
    """

    def _handle_batch(self, batch):
        # output is always tuple in hf
        # also if model is sepcified with task
        # loss is stored in outpt[0]
        loss = self.model(**batch)[0]

        if is_wrapped_with_ddp(self.model):
            loss = loss.mean()

        self.state.batch_metrics["loss"] = loss
