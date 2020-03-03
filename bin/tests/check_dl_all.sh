#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


####################################  DL  #####################################
bash ./bin/tests/check_dl_base.sh


####################################  CV  #####################################
bash ./bin/tests/check_dl_cv.sh


####################################  GAN  ####################################
if [[ "$CUDA_VISIBLE_DEVICES" == "" || "$CUDA_VISIBLE_DEVICES" == "0" ]]; then
    bash ./bin/tests/check_dl_gan.sh
fi


####################################  NLP  ####################################
bash ./bin/tests/check_dl_nlp.sh
