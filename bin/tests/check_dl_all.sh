#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


####################################  DL  #####################################
bash ./bin/tests/check_dl_base.sh


####################################  CV  #####################################
bash ./bin/tests/check_dl_cv.sh


####################################  NLP  ####################################
bash ./bin/tests/check_dl_nlp.sh
