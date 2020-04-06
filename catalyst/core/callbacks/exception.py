from catalyst.core import Callback, CallbackNode, CallbackOrder, State, utils


class ExceptionCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            order=CallbackOrder.External + 1, node=CallbackNode.All
        )

    def on_exception(self, state: State):
        """@TODO: Docs. Contribution is welcome."""
        exception = state.exception
        if not utils.is_exception(exception):
            return

        if state.need_exception_reraise:
            raise exception
