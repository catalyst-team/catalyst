from catalyst.__version__ import __version__  # noqa: F401

import warnings

warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed", append=False
)
warnings.filterwarnings("once", append=True)
