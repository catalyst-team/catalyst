#!/bin/bash
set -e -o xtrace

pip install -r requirements/requirements.txt
pip install -r requirements/requirements-cv.txt
pip install -r requirements/requirements-nlp.txt
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
  rm -rf *
  cp -a $TEMP/builds/* .

  git config --global user.email "teamcity@catalyst.github"
  git config --global user.name "Teamcity"
  git add .
  git commit -m "$COMMENT"  || true
  git push || true
fi