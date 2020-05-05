#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

pip install -r requirements/requirements.txt

###########################  check [catalyst-core]  ###########################
cat <<EOT > .catalyst
[catalyst]
contrib_required = false
cv_required = false
ml_required = false
nlp_required = false
EOT

python -c """
from catalyst.contrib.dl import callbacks
from catalyst.contrib import utils

try:
    callbacks.AlchemyLogger
except ImportError:
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' expected')
"""


##########################  check [catalyst-contrib]  #########################
cat <<EOT > .catalyst
[catalyst]
contrib_required = true
cv_required = false
ml_required = false
nlp_required = false
EOT

# fail if requirements not installed
python -c """
from catalyst.tools import settings

assert settings.use_lz4 == False and settings.use_pyarrow == False

try:
    from catalyst.contrib.dl.callbacks import AlchemyLogger, VisdomLogger
except ImportError:
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' expected')
"""

pip install -r requirements/requirements-contrib.txt
pip install -r requirements/requirements-ecosystem.txt
python -c """
from catalyst.contrib.dl.callbacks import AlchemyLogger, VisdomLogger
"""


############################  check [catalyst-cv]  ############################
cat <<EOT > .catalyst
[catalyst]
contrib_required = false
cv_required = true
ml_required = false
nlp_required = false
EOT

# fail if requirements not installed
python -c """
from catalyst.tools import settings

assert settings.use_libjpeg_turbo == False

try:
    from catalyst.contrib.data import cv as cv_data
    from catalyst.contrib.dl.callbacks import InferMaskCallback
    from catalyst.contrib.models import cv as cv_models
    from catalyst.contrib.utils import imread, imwrite
    from catalyst.data.__main__ import COMMANDS as DATA_SCRIPTS

    assert not (
        'process-images' in DATA_SCRIPTS
        or 'process-images' in DATA_SCRIPTS
        or 'project-embeddings' in DATA_SCRIPTS
    )
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-cv.txt
python -c """
from catalyst.contrib.data import cv as cv_data
from catalyst.contrib.dl.callbacks import InferMaskCallback
from catalyst.contrib.models import cv as cv_models
from catalyst.contrib.utils import imread, imwrite
from catalyst.data.__main__ import COMMANDS as DATA_SCRIPTS

assert (
    'process-images' in DATA_SCRIPTS
    and 'process-images' in DATA_SCRIPTS
    and 'project-embeddings' in DATA_SCRIPTS
)
"""


############################  check [catalyst-ml]  ############################
cat <<EOT > .catalyst
[catalyst]
contrib_required = false
cv_required = false
ml_required = true
nlp_required = false
EOT

# fail if requirements not installed
python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not (
        'check-index-model' in COMMANDS or 'create-index-model' in COMMANDS
    )
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-ml.txt
python -c """
from catalyst.contrib.__main__ import COMMANDS

assert 'check-index-model' in COMMANDS and 'create-index-model' in COMMANDS
"""


############################  check [catalyst-nlp]  ###########################
cat <<EOT > .catalyst
[catalyst]
contrib_required = false
cv_required = false
ml_required = false
nlp_required = true
EOT

# fail if requirements not installed
python -c """
try:
    from catalyst.contrib.data import nlp as nlp_data
    from catalyst.contrib.models import nlp as nlp_models
    from catalyst.contrib.utils import tokenize_text, process_bert_output
    from catalyst.contrib.__main__ import COMMANDS as CONTRIB_SCRIPTS
    from catalyst.data.__main__ import COMMANDS as DATA_SCRIPTS

    assert 'text2embedding' not in CONTRIB_SCRIPTS
    assert 'text2embedding' not in DATA_SCRIPTS
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-nlp.txt
python -c """
from catalyst.contrib.data import nlp as nlp_data
from catalyst.contrib.models import nlp as nlp_models
from catalyst.contrib.utils import tokenize_text, process_bert_output
from catalyst.contrib.__main__ import COMMANDS as CONTRIB_SCRIPTS
from catalyst.data.__main__ import COMMANDS as DATA_SCRIPTS

assert 'text2embedding' in CONTRIB_SCRIPTS
assert 'text2embedding' in DATA_SCRIPTS
"""


################################  pipeline 99  ################################
rm .catalyst
