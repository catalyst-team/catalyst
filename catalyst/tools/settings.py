# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Any, Dict, List, Optional, Tuple
import configparser
import logging
import os

from catalyst.tools.frozen_class import FrozenClass

logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm

    IS_XLA_AVAILABLE = True
except ModuleNotFoundError:
    IS_XLA_AVAILABLE = False

try:
    import torch.nn.utils.prune as prune

    IS_PRUNING_AVAILABLE = True
except ModuleNotFoundError:
    IS_PRUNING_AVAILABLE = False

try:
    from git import Repo as repo

    IS_GIT_AVAILABLE = True
except ImportError:
    IS_GIT_AVAILABLE = False

try:
    import torch.quantization

    IS_QUANTIZATION_AVAILABLE = True
except ModuleNotFoundError:
    IS_QUANTIZATION_AVAILABLE = False


class Settings(FrozenClass):
    def __init__(
        self,
        contrib_required: bool = False,
        cv_required: bool = False,
        ml_required: bool = False,
        nlp_required: bool = False,
        alchemy_logger_required: Optional[bool] = None,
        neptune_logger_required: Optional[bool] = None,
        visdom_logger_required: Optional[bool] = None,
        wandb_logger_required: Optional[bool] = None,
        optuna_required: Optional[bool] = None,
        plotly_required: Optional[bool] = None,
        telegram_logger_token: Optional[str] = None,
        telegram_logger_chat_id: Optional[str] = None,
        use_lz4: bool = False,
        use_pyarrow: bool = False,
        albumentations_required: Optional[bool] = None,
        kornia_required: Optional[bool] = None,
        segmentation_models_required: Optional[bool] = None,
        use_libjpeg_turbo: bool = False,
        nmslib_required: Optional[bool] = None,
        transformers_required: Optional[bool] = None,
    ):
        # [catalyst]
        self.contrib_required: bool = contrib_required
        self.cv_required: bool = cv_required
        self.ml_required: bool = ml_required
        self.nlp_required: bool = nlp_required

        # stages
        self.stage_train_prefix: str = "train"
        self.stage_valid_prefix: str = "valid"
        self.stage_infer_prefix: str = "infer"

        # loader
        self.loader_train_prefix: str = "train"
        self.loader_valid_prefix: str = "valid"
        self.loader_infer_prefix: str = "infer"

        # [catalyst-contrib]
        self.alchemy_logger_required: bool = self._optional_value(
            alchemy_logger_required, default=contrib_required
        )
        self.neptune_logger_required: bool = self._optional_value(
            neptune_logger_required, default=contrib_required
        )
        self.visdom_logger_required: bool = self._optional_value(
            visdom_logger_required, default=contrib_required
        )
        self.wandb_logger_required: bool = self._optional_value(
            wandb_logger_required, default=contrib_required
        )
        self.optuna_required: bool = self._optional_value(
            optuna_required, default=contrib_required
        )
        self.plotly_required: bool = self._optional_value(
            plotly_required, default=contrib_required
        )
        self.telegram_logger_token: str = telegram_logger_token
        self.telegram_logger_chat_id: str = telegram_logger_chat_id
        self.use_lz4: bool = use_lz4
        self.use_pyarrow: bool = use_pyarrow

        # [catalyst-cv]
        self.albumentations_required: bool = self._optional_value(
            albumentations_required, default=cv_required
        )
        self.kornia_required: bool = self._optional_value(
            kornia_required, default=cv_required
        )
        self.segmentation_models_required: bool = self._optional_value(
            segmentation_models_required, default=cv_required
        )
        self.use_libjpeg_turbo: bool = use_libjpeg_turbo

        # [catalyst-ml]
        self.nmslib_required: bool = self._optional_value(
            nmslib_required, default=ml_required
        )

        # [catalyst-nlp]
        self.transformers_required: bool = self._optional_value(
            transformers_required, default=nlp_required
        )

    @staticmethod
    def _optional_value(value, default):
        return value if value is not None else default

    def type_hint(self, key: str):
        # return get_type_hints(self).get(key, None)
        return type(getattr(self, key, None))

    @staticmethod
    def parse() -> "Settings":
        kwargrs = MergedConfigParser(ConfigFileFinder("catalyst")).parse()
        return Settings(**kwargrs)


default_settings = Settings()


class ConfigFileFinder:
    """Encapsulate the logic for finding and reading config files.

    Adapted from:

    - https://gitlab.com/pwoolvett/flake8 (MIT License)
    - https://github.com/python/mypy (MIT License)
    """

    def __init__(self, program_name: str) -> None:
        """Initialize object to find config files.

        Args:
            program_name (str): Name of the current program (e.g., catalyst).
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
            home_dir = os.environ.get(
                "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
            )
            config_file_basename = program_name

        return os.path.join(home_dir, config_file_basename)

    @staticmethod
    def _read_config(
        *files: str,
    ) -> Tuple[configparser.RawConfigParser, List[str]]:
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
                filename = os.path.abspath(
                    os.path.join(parent, project_filename)
                )
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
            config_finder (ConfigFileFinder): Initialized ConfigFileFinder.
        """
        self.program_name = config_finder.program_name
        self.config_finder = config_finder

    def _normalize_value(self, option, value):
        final_value = option.normalize(
            value, self.config_finder.local_directory
        )
        logger.debug(
            f"{value} has been normalized to {final_value}"
            f" for option '{option.config_name}'",
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
                type_ = default_settings.type_hint(option_name)
                method = type2method.get(type_, config_parser.get)
                config_dict[option_name] = method(
                    self.program_name, option_name
                )

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


settings = Settings.parse()


__all__ = [
    "settings",
    "Settings",
    "ConfigFileFinder",
    "MergedConfigParser",
    "IS_PRUNING_AVAILABLE",
    "IS_XLA_AVAILABLE",
    "IS_GIT_AVAILABLE",
    "IS_QUANTIZATION_AVAILABLE",
]
