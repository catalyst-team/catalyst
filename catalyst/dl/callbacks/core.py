class Callback:
    """
    An abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.

    usage example:

    stage start
        epoch start (one epoch - one run of every loader)
            loader start
                batch start
                batch handler
                batch end
            loader end
        epoch end
    stage end
    """

    def on_stage_start(self, state):
        pass

    def on_stage_end(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_loader_start(self, state):
        pass

    def on_loader_end(self, state):
        pass

    def on_batch_start(self, state):
        pass

    def on_batch_end(self, state):
        pass
