# flake8: noqa
import logging

from catalyst.tools import settings

logger = logging.getLogger(__name__)

try:
    from catalyst.data.nlp.dataset.language_modeling import (
        LanguageModelingDataset,
    )
    from catalyst.data.nlp.dataset.text_classification import (
        TextClassificationDataset,
    )
except ImportError as ex:
    if settings.nlp_required:
        logger.warning(
            "some of catalyst-nlp dependencies not available,"
            " to install dependencies, run `pip install catalyst[nlp]`."
        )
        raise ex
