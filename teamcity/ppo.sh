echo "apt update && apt install -y redis-server"
apt update && apt install -y redis-server

echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-rl.txt"
pip install -r requirements/requirements-rl.txt

echo "./bin/tests/check_rl_ppo.sh"
OMP_NUM_THREADS="1" MKL_NUM_THREADS="1" bash ./bin/tests/check_rl_ppo.sh