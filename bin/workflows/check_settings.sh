#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

function clean_requirements() {
  pip uninstall -r requirements/requirements-cv.txt -y
  pip uninstall -r requirements/requirements-dev.txt -y
  pip uninstall -r requirements/requirements-hydra.txt -y
  pip uninstall -r requirements/requirements-ml.txt -y
  pip uninstall -r requirements/requirements-optuna.txt -y
  pip uninstall -r requirements/requirements-mlflow.txt -y
  pip uninstall -r requirements/requirements-neptune.txt -y
  pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed
}

function require_libs() {
  # usage: require_libs ml neptune

  # base .catalyst file when no parameters are given
  CONFIG="$(echo "
[catalyst]
cv_required = false
ml_required = false
hydra_required = false
optuna_required = false
mlflow_required = false
neptune_required = false
")"
  for REQUIRED in "$@"
  do
    CONFIG="$(echo "$CONFIG" | sed -re "s/^($REQUIRED)_required\\s*=\\s*false/\\1_required = true/gi")"
  done
  echo "$CONFIG" > .catalyst
}

################################  pipeline 00  ################################
# checking catalyst-core loading
clean_requirements
require_libs

cat <<EOT > .catalyst
[catalyst]
cv_required = false
ml_required = false
hydra_required = false
optuna_required = false
mlflow_required = false
EOT

python -c """
from catalyst.contrib import utils

try:
    utils.imread
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""


################################  pipeline 01  ################################
# checking catalyst-cv dependencies loading
clean_requirements
require_libs cv

python -c """
try:
    from catalyst.contrib.dataset import cv as cv_data
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.models import cv as cv_models
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.utils import imread, imwrite
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not ('process-images' in COMMANDS)
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""


pip install -r requirements/requirements-cv.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.contrib.data import cv as cv_data
from catalyst.contrib.models import cv as cv_models
from catalyst.contrib.utils import imread, imwrite
"""


################################  pipeline 02  ################################
# checking catalyst-ml dependencies loading
clean_requirements
require_libs ml

# check if fail if requirements not installed
python -c """
try:
    from catalyst.contrib.utils import balance_classes, split_dataframe_train_test
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not (
        'tag2label' in COMMANDS
        or 'split-dataframe' in COMMANDS
    )
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-ml.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.contrib.utils import balance_classes, split_dataframe_train_test
from catalyst.contrib.__main__ import COMMANDS

assert (
    'tag2label' in COMMANDS
    and 'split-dataframe' in COMMANDS
)
"""

################################  pipeline 03  ################################
# checking catalyst-(cv/ml) dependencies loading
clean_requirements
require_libs cv ml

cat <<EOT > .catalyst
[catalyst]
cv_required = true
ml_required = true
hydra_required = false
mlflow_required = false
neptune_required = false
EOT

python -c """
try:
    from catalyst.contrib.dataset import cv as cv_data
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.models import cv as cv_models
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.utils import imread, imwrite
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not ('process-images' in COMMANDS)
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.utils import balance_classes, split_dataframe_train_test
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not (
        'tag2label' in COMMANDS
        or 'split-dataframe' in COMMANDS
    )
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""


pip install -r requirements/requirements-cv.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

pip install -r requirements/requirements-ml.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.contrib.data import cv as cv_data
from catalyst.contrib.models import cv as cv_models
from catalyst.contrib.utils import imread, imwrite
from catalyst.contrib.__main__ import COMMANDS

assert 'process-images' in COMMANDS
"""

python -c """
from catalyst.contrib.utils import balance_classes, split_dataframe_train_test
from catalyst.contrib.__main__ import COMMANDS

assert (
    'tag2label' in COMMANDS
    and 'split-dataframe' in COMMANDS
)
"""

################################  pipeline 04  ################################
# checking catalyst-hydra dependencies loading
clean_requirements
require_libs hydra

# check if fail if requirements not installed
python -c """
try:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-hydra.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner
"""

################################  pipeline 05  ################################
# checking catalyst-optuna dependencies loading
clean_requirements
require_libs optuna

# check if fail if requirements not installed
python -c """
try:
    from catalyst.dl.__main__ import COMMANDS

    assert not ('tune' in COMMANDS)
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-optuna.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.dl.__main__ import COMMANDS

assert ('tune' in COMMANDS)
"""

################################  pipeline 06  ################################
# checking catalyst-mlflow dependencies loading
clean_requirements
require_libs mlflow

# check if fail if requirements not installed
python -c """
try:
    from catalyst.loggers import MLflowLogger
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-mlflow.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.loggers import MLflowLogger
"""

################################  pipeline 06  ################################
# checking catalyst-neptune dependencies loading
clean_requirements
require_libs neptune

# check if fail if requirements not installed
python -c """
try:
    from catalyst.loggers import NeptuneLogger
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

pip install -r requirements/requirements-neptune.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.loggers import NeptuneLogger
"""

################################  pipeline 99  ################################
rm .catalyst
