# flake8: noqa

import logging
import os

from catalyst.utils import *
# from .trace import get_trace_name, load_traced_model, trace_model
from .pipelines import clone_pipeline
from .torch import get_loader
from .visualization import plot_metrics
from .wizard import run_wizard, Wizard

logger = logging.getLogger(__name__)


try:
    import transformers  # noqa: F401
    from .text import tokenize_text, process_bert_output
except ImportError as ex:
    if os.environ.get("USE_TRANSFORMERS", "0") == "1":
        logger.warning(
            "transformers not available, to install transformers,"
            " run `pip install transformers`."
        )
        raise ex
