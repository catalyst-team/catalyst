#!/bin/bash
set -e -o xtrace

pip install \
    -r ./requirements/requirements.txt \
    -r ./requirements/requirements-dev.txt \
    -r ./requirements/requirements-ml.txt \
    -r ./requirements/requirements-cv.txt \
    -r ./requirements/requirements-nlp.txt \
    -r ./requirements/requirements-contrib.txt
pip install -r docs/requirements.txt

# @TODO: fix server issue
pip install torch==1.4.0 torchvision==0.5.0

###################################  DOCS  ####################################

REMOVE_BUILDS=0 make check-docs

COMMENT=$(git log -1 --pretty=%B)

cp -a builds $TEMP/builds

if [ $GIT_BRANCH == 'refs/heads/master' ]; then
  cd $TEMP

  git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/catalyst.git

  cd catalyst
  # Remove master docs, do not touch the past versions 
  rm -f *
  rm -rf .doctrees _modules _sources _static api info
  cp -a $TEMP/builds/* .

  git config --global user.email "teamcity@catalyst.github"
  git config --global user.name "Teamcity"
  git add .
  git commit -m "$COMMENT"  || true
  git push || true
fi
