#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


###################################  APEX  ####################################
if [[ -d /mount/apex ]]; then
  rm -rf /tmp/apex
  cp -a /mount/apex /tmp/apex
  pip install -v --no-cache-dir \
      -e /tmp/apex \
      --global-option="--cpp_ext" --global-option="--cuda_ext"
else
  pip install -v --no-cache-dir \
      --global-option="--cpp_ext" --global-option="--cuda_ext" \
      git+https://github.com/NVIDIA/apex
fi
