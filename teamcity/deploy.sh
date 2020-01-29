set -e

echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-cv.txt"
pip install -r requirements/requirements-cv.txt

echo "pip install -r requirements/requirements-nlp.txt"
pip install -r requirements/requirements-nlp.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt

echo "pip install -r docs/requirements.txt"
pip install -r docs/requirements.txt

echo "REMOVE_BUILDS=0 make check-docs"
REMOVE_BUILDS=0 make check-docs

echo "COMMENT=$(git log -1 --pretty=%B)"
COMMENT=$(git log -1 --pretty=%B)

echo "cp -a builds $TEMP/builds"
cp -a builds $TEMP/builds

echo "cd $TEMP"
cd $TEMP

echo "git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/catalyst.git"
git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/catalyst.git

echo "copying files"
cd catalyst
rm -rf *
cp -a $TEMP/builds/* .

echo "git commit and push"
git config --global user.email "teamcity@catalyst.github"
git config --global user.name "Teamcity"
git add .
git commit -m "$COMMENT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ $BRANCH == 'master' ]; then
  git push
fi