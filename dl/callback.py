from typing import Dict


class Callback:
    """
    An abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.

    usage example:

    mode start (train/infer/debug)
        epoch start (one epoch - one run of every loader)
            loader start
                batch start
                batch handler
                batch end
            loader end
        epoch end
    mode end
    """

    def on_train_start(self, state): pass

    def on_train_end(self, state): pass

    def on_infer_start(self, state): pass

    def on_infer_end(self, state): pass

    def on_epoch_start(self, state): pass

    def on_epoch_end(self, state): pass

    def on_loader_start(self, state): pass

    def on_loader_end(self, state): pass

    def on_batch_start(self, state): pass

    def on_batch_end(self, state): pass


class CallbackCompose:

    def __init__(self, callbacks: Dict[str, Callback]):
        self.callbacks = callbacks

    def on_train_start(self, state):
        for key, value in self.callbacks.items():
            value.on_train_start(state=state)

    def on_train_end(self, state):
        for key, value in self.callbacks.items():
            value.on_train_end(state=state)

    def on_infer_start(self, state):
        for key, value in self.callbacks.items():
            value.on_infer_start(state=state)

    def on_infer_end(self, state):
        for key, value in self.callbacks.items():
            value.on_infer_end(state=state)

    def on_epoch_start(self, state):
        for key, value in self.callbacks.items():
            value.on_epoch_start(state=state)

    def on_epoch_end(self, state):
        for key, value in self.callbacks.items():
            value.on_epoch_end(state=state)

    def on_loader_start(self, state):
        for key, value in self.callbacks.items():
            value.on_loader_start(state=state)

    def on_loader_end(self, state):
        for key, value in self.callbacks.items():
            value.on_loader_end(state=state)

    def on_batch_start(self, state):
        for key, value in self.callbacks.items():
            value.on_batch_start(state=state)

    def on_batch_end(self, state):
        for key, value in self.callbacks.items():
            value.on_batch_end(state=state)
