echo 'apt-get update && apt-get install wget'
apt-get update && apt-get install wget

echo 'pip install -r requirements/requirements.txt'
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-cv.txt"
pip install -r requirements/requirements-cv.txt

echo 'pip install -r requirements/requirements-nlp.txt'
pip install -r requirements/requirements-nlp.txt

echo 'pip install -r requirements/requirements-rl.txt'
pip install -r requirements/requirements-rl.txt

echo './bin/tests/check_dl.sh'
./bin/tests/check_dl.sh

echo './bin/tests/check_nlp.sh'
./bin/tests/check_nlp.sh
