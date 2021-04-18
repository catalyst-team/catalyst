# flake8: noqa
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed", append=True)
warnings.filterwarnings("ignore", module="tqdm", append=True)
warnings.filterwarnings("once", append=True)

from catalyst.__version__ import __version__
from catalyst.settings import SETTINGS
