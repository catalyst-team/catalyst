from catalyst import utils
from catalyst.core import Callback, CallbackNode, CallbackOrder, State


class ExceptionCallback(Callback):
    def __init__(self):
        super().__init__(
            order=CallbackOrder.External + 1, node=CallbackNode.All
        )

    def on_exception(self, state: State):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if state.need_exception_reraise:
            raise exception
