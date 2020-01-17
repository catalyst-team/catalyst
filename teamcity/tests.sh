echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-nlp.txt"
pip install -r requirements/requirements-nlp.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt

echo "pip install -r requirements/requirements-dev.txt"
pip install -r requirements/requirements-dev.txt

echo "isort -rc --check-only --settings-path ./setup.cfg"
isort -rc --check-only --settings-path ./setup.cfg

sleep 100000

echo "make check-style"
make check-style