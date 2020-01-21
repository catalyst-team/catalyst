echo "apt update && apt install -y redis-server"
apt update && apt install -y redis-server

echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt

echo "./bin/tests/check_reinforce.sh"
./bin/tests/check_reinforce.sh