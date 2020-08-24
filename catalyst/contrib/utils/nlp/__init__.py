# flake8: noqa

import logging

logger = logging.getLogger(__name__)

from catalyst.tools import settings

try:
    import transformers  # noqa: F401

    from catalyst.contrib.utils.nlp.text import (
        process_bert_output,
        tokenize_text,
    )
except ImportError as ex:
    if settings.transformers_required:
        logger.warning(
            "transformers not available, to install transformers,"
            " run `pip install transformers`."
        )
        raise ex
