# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class ExceptionCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            order=CallbackOrder.external + 1, node=CallbackNode.all
        )

    def on_exception(self, runner: IRunner):
        """@TODO: Docs. Contribution is welcome."""
        exception = runner.exception
        if not utils.is_exception(exception):
            return

        if runner.need_exception_reraise:
            raise exception


__all__ = ["ExceptionCallback"]
