echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-cv.txt"
pip install -r requirements/requirements-cv.txt

echo "pip install -r requirements/requirements-nlp.txt"
pip install -r requirements/requirements-nlp.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt

echo "pip install -r requirements/requirements-dev.txt"
pip install -r requirements/requirements-dev.txt

echo "make check-codestyle"
make check-codestyle
