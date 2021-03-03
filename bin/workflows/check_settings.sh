#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  pipeline 00  ################################
# checking catalyst-core loading
pip uninstall -r requirements/requirements-cv.txt -y
pip uninstall -r requirements/requirements-dev.txt -y
pip uninstall -r requirements/requirements-hydra.txt -y
pip uninstall -r requirements/requirements-ml.txt -y
pip uninstall -r requirements/requirements-optuna.txt -y
pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

cat <<EOT > .catalyst
[catalyst]
cv_required = false
ml_required = false
hydra_required = false
optuna_required = false
EOT

python -c """
from catalyst.contrib import utils

try:
    utils.imread
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""


################################  pipeline 01  ################################
# checking catalyst-cv dependencies loading
pip uninstall -r requirements/requirements-cv.txt -y
pip uninstall -r requirements/requirements-dev.txt -y
pip uninstall -r requirements/requirements-hydra.txt -y
pip uninstall -r requirements/requirements-ml.txt -y
pip uninstall -r requirements/requirements-optuna.txt -y
pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

cat <<EOT > .catalyst
[catalyst]
cv_required = true
ml_required = false
hydra_required = false
optuna_required = false
EOT

# check if fail if requirements not installed
python -c """
from catalyst.settings import SETTINGS

assert SETTINGS.use_libjpeg_turbo == False
"""

python -c """
try:
    from catalyst.contrib.dataset import cv as cv_data
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.models import cv as cv_models
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.utils import imread, imwrite
except (ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

python -c """
try:
    from catalyst.contrib.__main__ import COMMANDS

    assert not ('process-images' in COMMANDS)
except (ImportError, AssertionError):
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
from catalyst.contrib.__main__ import COMMANDS

assert 'process-images' in COMMANDS
"""


################################  pipeline 02  ################################
# checking catalyst-ml dependencies loading
pip uninstall -r requirements/requirements-cv.txt -y
pip uninstall -r requirements/requirements-dev.txt -y
pip uninstall -r requirements/requirements-hydra.txt -y
pip uninstall -r requirements/requirements-ml.txt -y
pip uninstall -r requirements/requirements-optuna.txt -y
pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

cat <<EOT > .catalyst
[catalyst]
cv_required = false
ml_required = true
hydra_required = false
optuna_required = false
EOT

# check if fail if requirements not installed
python -c """
try:
    from catalyst.contrib.utils import balance_classes, split_dataframe_train_test
except (ImportError, AssertionError):
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
except (ImportError, AssertionError):
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
# checking catalyst-hydra dependencies loading
pip uninstall -r requirements/requirements-cv.txt -y
pip uninstall -r requirements/requirements-dev.txt -y
pip uninstall -r requirements/requirements-hydra.txt -y
pip uninstall -r requirements/requirements-ml.txt -y
pip uninstall -r requirements/requirements-optuna.txt -y
pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

cat <<EOT > .catalyst
[catalyst]
cv_required = false
ml_required = false
hydra_required = true
optuna_required = false
EOT

# check if fail if requirements not installed
python -c """
try:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner
except (ImportError, AssertionError):
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

################################  pipeline 04  ################################
# checking catalyst-optuna dependencies loading
pip uninstall -r requirements/requirements-cv.txt -y
pip uninstall -r requirements/requirements-dev.txt -y
pip uninstall -r requirements/requirements-hydra.txt -y
pip uninstall -r requirements/requirements-ml.txt -y
pip uninstall -r requirements/requirements-optuna.txt -y
pip install -r requirements/requirements.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

cat <<EOT > .catalyst
[catalyst]
cv_required = false
ml_required = false
hydra_required = false
optuna_required = true
EOT

# check if fail if requirements not installed
python -c """
from catalyst.dl.__main__ import COMMANDS

assert not ('tune' in COMMANDS)
"""

pip install -r requirements/requirements-optuna.txt --quiet \
  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
  --upgrade-strategy only-if-needed

python -c """
from catalyst.dl.__main__ import COMMANDS

assert ('tune' in COMMANDS)
"""


################################  pipeline 99  ################################
rm .catalyst
