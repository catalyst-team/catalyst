from catalyst.__version__ import __version__  # noqa: F401

import warnings

warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed", append=True
)
warnings.filterwarnings("ignore", module="tqdm", append=True)
warnings.filterwarnings("once", append=True)
