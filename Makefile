.PHONY: check-style check-docs docker docker-dev clean

check-style:
	bash ./bin/_check_codestyle.sh -s

check-docs:
	bash ./bin/_check_docs.sh

docker: ./requirements/requirements.txt ./requirements/requirements-rl.txt
	docker build -t catalyst-base:latest . -f ./docker/Dockerfile

docker-dev: ./requirements/requirements.txt ./requirements/requirements-rl.txt ./requirements/requirements-dev.txt
	docker build -t catalyst-dev:latest . -f ./docker/Dockerfile-dev

install-from-source:
	pip uninstall catalyst -y && pip install -e ./

clean:
	rm -rf build/
	docker rmi -f catalyst-base:latest
	docker rmi -f catalyst-dev:latest
