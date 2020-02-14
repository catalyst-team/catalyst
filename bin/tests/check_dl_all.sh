#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


####################################  DL  #####################################
echo './bin/tests/check_dl_base.sh'
bash ./bin/tests/check_dl_base.sh


####################################  CV  #####################################
echo './bin/tests/check_dl_cv.sh'
bash ./bin/tests/check_dl_cv.sh


####################################  GAN  ####################################
echo './bin/tests/check_dl_gan.sh'
bash ./bin/tests/check_dl_gan.sh


####################################  NLP  ####################################
echo './bin/tests/check_dl_nlp.sh'
bash ./bin/tests/check_dl_nlp.sh
