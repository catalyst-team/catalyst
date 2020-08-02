# flake8: noqa
import warnings


class IRunnerLegacy:
    """
    Special class to encapsulate all `catalyst.core.runner.IRunner`
    and `catalyst.core.runner.State` legacy into one place.
    Used to make `catalyst.core.runner.IRunner` cleaner
    and easier to understand.

    Saved for backward compatibility. Should be removed someday.
    """

    @property
    def batch_in(self):
        """Alias for `runner.input`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `runner.input` instead.
        """
        warnings.warn(
            "`runner.batch_in` was deprecated, "
            "please use `runner.input` instead",
            DeprecationWarning,
        )
        return self.input

    @property
    def batch_out(self):
        """Alias for `runner.output`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `runner.output` instead.
        """
        warnings.warn(
            "`runner.batch_out` was deprecated, "
            "please use `runner.output` instead",
            DeprecationWarning,
        )
        return self.output

    @property
    def need_backward_pass(self):
        """Alias for `runner.is_train_loader`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `runner.is_train_loader` instead.
        """
        warnings.warn(
            "`need_backward_pass` was deprecated, "
            "please use `is_train_loader` instead",
            DeprecationWarning,
        )
        return self.is_train_loader

    @property
    def loader_step(self):
        """Alias for `runner.loader_batch_step`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `runner.loader_batch_step` instead.
        """
        warnings.warn(
            "`loader_step` was deprecated, "
            "please use `loader_batch_step` instead",
            DeprecationWarning,
        )
        return self.loader_batch_step

    @property
    def state(self):
        """Alias for `runner`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `runner` instead.
        """
        warnings.warn(
            "`runner.state` was deprecated, " "please use `runner` instead",
            DeprecationWarning,
        )
        return self
