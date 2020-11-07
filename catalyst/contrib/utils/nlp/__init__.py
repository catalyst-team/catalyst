# flake8: noqa

import logging

logger = logging.getLogger(__name__)

from catalyst.settings import SETTINGS

try:
    import transformers  # noqa: F401
    from catalyst.contrib.utils.nlp.text import (
        tokenize_text,
        process_bert_output,
    )
except ImportError as ex:
    if SETTINGS.transformers_required:
        logger.warning(
            "transformers not available, to install transformers,"
            " run `pip install transformers`."
        )
        raise ex
