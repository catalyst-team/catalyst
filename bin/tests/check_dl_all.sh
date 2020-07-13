#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  CORE  ####################################
(set -e; for f in ./bin/tests/check_dl_core*.sh; do bash "$f"; done)


####################################  CV  #####################################
(set -e; for f in ./bin/tests/check_dl_cv*.sh; do bash "$f"; done)


####################################  NLP  ####################################
(set -e; for f in ./bin/tests/check_dl_nlp*.sh; do bash "$f"; done)

################################  Metric Learning  #############################
(set -e; for f in ./bin/tests/check_dl_ml*.sh; do bash "$f"; done)