from typing import TYPE_CHECKING

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.utils.misc import is_exception

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ExceptionCallback(Callback):
    """Handles python exceptions during run."""

    def __init__(self):
        """Initialisation for ExceptionCallback."""
        super().__init__(
            order=CallbackOrder.external + 1, node=CallbackNode.all
        )

    def on_exception(self, runner: "IRunner") -> None:
        """Exception handle hook.  # noqa: DAR401

        Args:
            runner: experiment runner

        Raises:
            Exception: if during exception handling,  # noqa: DAR402
                we still need to reraise exception
        """
        exception = runner.exception
        if not is_exception(exception):
            return

        if runner.need_exception_reraise:
            raise exception


__all__ = ["ExceptionCallback"]
