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
if [[ "$CUDA_VISIBLE_DEVICES" == "" ] || [ "$CUDA_VISIBLE_DEVICES" == "0" ]]; then
    echo './bin/tests/check_dl_gan.sh'
    bash ./bin/tests/check_dl_gan.sh
fi


####################################  NLP  ####################################
echo './bin/tests/check_dl_nlp.sh'
bash ./bin/tests/check_dl_nlp.sh
