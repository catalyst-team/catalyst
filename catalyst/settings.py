from typing import Any, Dict, List, Optional, Tuple
import configparser
import logging
import os

import torch

from catalyst.tools.frozen_class import FrozenClass

logger = logging.getLogger(__name__)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
NUM_CUDA_DEVICES = torch.cuda.device_count()


try:
    from git import Repo  # noqa: F401

    IS_GIT_AVAILABLE = True
except ImportError:
    IS_GIT_AVAILABLE = False

try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    IS_XLA_AVAILABLE = True
except ModuleNotFoundError:
    IS_XLA_AVAILABLE = False

try:
    import torch.nn.utils.prune as prune  # noqa: F401

    IS_PRUNING_AVAILABLE = True
except ModuleNotFoundError:
    IS_PRUNING_AVAILABLE = False

try:
    import torch.quantization  # noqa: F401

    IS_QUANTIZATION_AVAILABLE = True
except ModuleNotFoundError:
    IS_QUANTIZATION_AVAILABLE = False

try:
    import optuna  # noqa: F401

    IS_OPTUNA_AVAILABLE = True
except ModuleNotFoundError:
    IS_OPTUNA_AVAILABLE = False

try:
    import hydra  # noqa: F401

    IS_HYDRA_AVAILABLE = True
except ModuleNotFoundError:
    IS_HYDRA_AVAILABLE = False


class Settings(FrozenClass):
    """Catalyst settings."""

    def __init__(  # noqa: D107
        self,
        # CV
        cv_required: bool = False,
        albumentations_required: Optional[bool] = None,
        kornia_required: Optional[bool] = None,
        segmentation_models_required: Optional[bool] = None,
        use_libjpeg_turbo: bool = False,
        # LOG
        log_required: bool = False,
        alchemy_required: Optional[bool] = None,
        neptune_required: Optional[bool] = None,
        wandb_required: Optional[bool] = None,
        plotly_required: Optional[bool] = None,
        # ML
        ml_required: bool = False,
        ipython_required: Optional[bool] = None,
        matplotlib_required: Optional[bool] = None,
        scipy_required: Optional[bool] = None,
        pandas_required: Optional[bool] = None,
        sklearn_required: Optional[bool] = None,
        git_required: Optional[bool] = None,
        # NLP
        nlp_required: bool = False,
        transformers_required: Optional[bool] = None,
        # TUNE
        tune_required: bool = False,
        optuna_required: Optional[bool] = None,
        # KNN
        nmslib_required: Optional[bool] = False,
        # extras
        use_lz4: bool = False,
        use_pyarrow: bool = False,
        telegram_logger_token: Optional[str] = None,
        telegram_logger_chat_id: Optional[str] = None,
        # HYDRA
        hydra_required: Optional[bool] = False,
    ):
        # [catalyst]
        self.cv_required: bool = cv_required
        self.log_required: bool = log_required
        self.ml_required: bool = ml_required
        self.nlp_required: bool = nlp_required
        self.tune_required: bool = tune_required

        # [catalyst-cv]
        self.albumentations_required: bool = self._optional_value(
            albumentations_required, default=cv_required
        )
        self.kornia_required: bool = self._optional_value(kornia_required, default=cv_required)
        self.segmentation_models_required: bool = self._optional_value(
            segmentation_models_required, default=cv_required
        )
        self.use_libjpeg_turbo: bool = use_libjpeg_turbo

        # [catalyst-log]
        self.alchemy_required: bool = self._optional_value(alchemy_required, default=log_required)
        self.neptune_required: bool = self._optional_value(neptune_required, default=log_required)
        self.wandb_required: bool = self._optional_value(wandb_required, default=log_required)
        self.plotly_required: bool = self._optional_value(plotly_required, default=log_required)

        # [catalyst-ml]
        self.scipy_required: bool = self._optional_value(scipy_required, default=ml_required)
        self.matplotlib_required: bool = self._optional_value(
            matplotlib_required, default=ml_required
        )
        self.pandas_required: bool = self._optional_value(pandas_required, default=ml_required)
        self.sklearn_required: bool = self._optional_value(sklearn_required, default=ml_required)
        self.ipython_required: bool = self._optional_value(ipython_required, default=ml_required)
        self.git_required: bool = self._optional_value(git_required, default=ml_required)

        # [catalyst-nlp]
        self.transformers_required: bool = self._optional_value(
            transformers_required, default=nlp_required
        )

        # [catalyst-tune]
        self.optuna_required: bool = self._optional_value(optuna_required, default=tune_required)

        # [catalyst-knn]
        self.nmslib_required: bool = nmslib_required

        # [catalyst-extras]
        self.use_lz4: bool = use_lz4
        self.use_pyarrow: bool = use_pyarrow
        self.telegram_logger_token: str = telegram_logger_token
        self.telegram_logger_chat_id: str = telegram_logger_chat_id

        # [catalyst-hydra]
        self.hydra_required: bool = hydra_required

        # [catalyst-global]
        # stages
        self.stage_train_prefix: str = "train"
        self.stage_valid_prefix: str = "valid"
        self.stage_infer_prefix: str = "infer"

        # epoch
        self.epoch_metrics_prefix: str = "_epoch_"

        # loader
        self.loader_train_prefix: str = "train"
        self.loader_valid_prefix: str = "valid"
        self.loader_infer_prefix: str = "infer"

    @staticmethod
    def _optional_value(value, default):
        return value if value is not None else default

    def type_hint(self, key: str):
        """Returns type hint for the specified ``key``.

        Args:
            key: key of interest

        Returns:
            type hint for the specified key
        """
        # return get_type_hints(self).get(key, None)
        return type(getattr(self, key, None))

    @staticmethod
    def parse() -> "Settings":
        """Parse and return the settings.

        Returns:
            Settings: Dictionary of the parsed and merged Settings.
        """
        kwargrs = MergedConfigParser(ConfigFileFinder("catalyst")).parse()
        return Settings(**kwargrs)


DEFAULT_SETTINGS = Settings()


class ConfigFileFinder:
    """Encapsulate the logic for finding and reading config files.

    Adapted from:

    - https://gitlab.com/pwoolvett/flake8 (MIT License)
    - https://github.com/python/mypy (MIT License)
    """

    def __init__(self, program_name: str) -> None:
        """Initialize object to find config files.

        Args:
            program_name: Name of the current program (e.g., catalyst).
        """
        # user configuration file
        self.program_name = program_name
        self.user_config_file = self._user_config_file(program_name)

        # list of filenames to find in the local/project directory
        self.project_filenames = ("setup.cfg", "tox.ini", f".{program_name}")

        self.local_directory = os.path.abspath(os.curdir)

    @staticmethod
    def _user_config_file(program_name: str) -> str:
        if os.name == "nt":  # if running on Windows
            home_dir = os.path.expanduser("~")
            config_file_basename = f".{program_name}"
        else:
            home_dir = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            config_file_basename = program_name

        return os.path.join(home_dir, config_file_basename)

    @staticmethod
    def _read_config(*files: str,) -> Tuple[configparser.RawConfigParser, List[str]]:
        config = configparser.RawConfigParser()

        found_files: List[str] = []
        for filename in files:
            try:
                found_files.extend(config.read(filename))
            except UnicodeDecodeError:
                logger.exception(
                    f"There was an error decoding a config file."
                    f" The file with a problem was {filename}."
                )
            except configparser.ParsingError:
                logger.exception(
                    f"There was an error trying to parse a config file."
                    f" The file with a problem was {filename}."
                )

        return config, found_files

    def generate_possible_local_files(self):
        """Find and generate all local config files.

        Yields:
            str: Path to config file.
        """
        parent = tail = os.getcwd()
        found_config_files = False
        while tail and not found_config_files:
            for project_filename in self.project_filenames:
                filename = os.path.abspath(os.path.join(parent, project_filename))
                if os.path.exists(filename):
                    yield filename
                    found_config_files = True
                    self.local_directory = parent
            (parent, tail) = os.path.split(parent)

    def local_config_files(self) -> List[str]:  # noqa: D202
        """
        Find all local config files which actually exist.

        Returns:
            List[str]: List of files that exist that are
            local project config  files with extra config files
            appended to that list (which also exist).
        """
        return list(self.generate_possible_local_files())

    def local_configs(self):
        """Parse all local config files into one config object."""
        config, found_files = self._read_config(*self.local_config_files())
        if found_files:
            logger.debug(f"Found local configuration files: {found_files}")
        return config

    def user_config(self):
        """Parse the user config file into a config object."""
        config, found_files = self._read_config(self.user_config_file)
        if found_files:
            logger.debug(f"Found user configuration files: {found_files}")
        return config


class MergedConfigParser:
    """Encapsulate merging different types of configuration files.

    This parses out the options registered that were specified in the
    configuration files, handles extra configuration files, and returns
    dictionaries with the parsed values.

    Adapted from:

    - https://gitlab.com/pwoolvett/flake8 (MIT License)
    - https://github.com/python/mypy (MIT License)
    """

    #: Set of actions that should use the
    #: :meth:`~configparser.RawConfigParser.getbool` method.
    GETBOOL_ACTIONS = {"store_true", "store_false"}  # noqa: WPS115

    def __init__(self, config_finder: ConfigFileFinder):
        """Initialize the MergedConfigParser instance.

        Args:
            config_finder: Initialized ConfigFileFinder.
        """
        self.program_name = config_finder.program_name
        self.config_finder = config_finder

    def _normalize_value(self, option, value):
        final_value = option.normalize(value, self.config_finder.local_directory)
        logger.debug(
            f"{value} has been normalized to {final_value}" f" for option '{option.config_name}'",
        )
        return final_value

    def _parse_config(self, config_parser):
        type2method = {
            bool: config_parser.getboolean,
            int: config_parser.getint,
        }

        config_dict: Dict[str, Any] = {}
        if config_parser.has_section(self.program_name):
            for option_name in config_parser.options(self.program_name):
                type_ = DEFAULT_SETTINGS.type_hint(option_name)
                method = type2method.get(type_, config_parser.get)
                config_dict[option_name] = method(self.program_name, option_name)

        return config_dict

    def parse(self) -> dict:
        """Parse and return the local and user config files.

        First this copies over the parsed local configuration and then
        iterates over the options in the user configuration and sets them if
        they were not set by the local configuration file.

        Returns:
            dict: Dictionary of the parsed and merged configuration options.
        """
        user_config = self._parse_config(self.config_finder.user_config())
        config = self._parse_config(self.config_finder.local_configs())

        for option, value in user_config.items():
            config.setdefault(option, value)

        return config


SETTINGS = Settings.parse()
setattr(SETTINGS, "IS_GIT_AVAILABLE", IS_GIT_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "IS_XLA_AVAILABLE", IS_XLA_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "IS_PRUNING_AVAILABLE", IS_PRUNING_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "IS_QUANTIZATION_AVAILABLE", IS_QUANTIZATION_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "IS_CUDA_AVAILABLE", IS_CUDA_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "NUM_CUDA_DEVICES", NUM_CUDA_DEVICES)  # noqa: B010
setattr(SETTINGS, "IS_OPTUNA_AVAILABLE", IS_OPTUNA_AVAILABLE)  # noqa: B010
setattr(SETTINGS, "IS_HYDRA_AVAILABLE", IS_HYDRA_AVAILABLE)  # noqa: B010


__all__ = [
    "SETTINGS",
    "Settings",
    "ConfigFileFinder",
    "MergedConfigParser",
    "IS_PRUNING_AVAILABLE",
    "IS_XLA_AVAILABLE",
    "IS_GIT_AVAILABLE",
    "IS_QUANTIZATION_AVAILABLE",
    "IS_OPTUNA_AVAILABLE",
    "IS_HYDRA_AVAILABLE",
]
