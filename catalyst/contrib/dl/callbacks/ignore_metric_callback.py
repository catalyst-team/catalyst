from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner


class IgnoreMetricCallback(Callback):
    """
    Ignore metric callbacks for specified loaders.
    """

    def __init__(self, **kwargs):
        """

        Args:
            kwargs: loader and callback names to ignore
        """
        super().__init__(order=CallbackOrder.External)
        # contains pointers to callbacks
        self.callbacks = {}
        self.loader_ignore_list = {}
        for loader, ignore_list in kwargs.items():
            if not isinstance(ignore_list, (str, list, tuple)):
                raise TypeError(
                    "Expected ignore list object is str/List[str]/Tuple[str] "
                    f"but got {type(ignore_list)}"
                )
            if isinstance(ignore_list, str):
                to_ignore = [ignore_list]
            else:
                to_ignore = [
                    str(callback_name) for callback_name in ignore_list
                ]
            self.loader_ignore_list[loader] = to_ignore

    def on_stage_start(self, runner: IRunner) -> None:
        """Get information about callbacks used in a stage.

        Args:
            runner (IRunner): current runner
        """
        for name, callback in runner.callbacks.items():
            self.callbacks[name] = callback

    def _is_correct_loader(
        self, loader: str, name: str, callback: Callback
    ) -> bool:
        """
        Check if callback should be used with loader.

        Args:
            loader (str): loader name
            name (str): callback name
            callback (Callback): callback object

        Returns:
            True if callback should be used with passed loader otherwise False
        """
        ignore_list = self.loader_ignore_list.get(loader) or []
        in_ignore_list = name in ignore_list
        is_metric = callback.order in (
            CallbackOrder.Metric,
            CallbackOrder.MetricAggregation,
        )
        return not (in_ignore_list and is_metric)

    def on_loader_start(self, runner: IRunner) -> None:
        """
        Construct list of callbacks for current loader.

        Args:
            runner (IRunner): current runner
        """
        loader = runner.loader_name
        filtered_loader_callbacks = {
            name: callback
            for name, callback in self.callbacks.items()
            if self._is_correct_loader(loader, name, callback)
        }
        runner.callbacks = filtered_loader_callbacks
