.PHONY: check-style check-docs docker docker-dev clean

check-style:
	bash ./bin/_check_codestyle.sh

check-docs:
	bash ./bin/_check_docs.sh

docker: requirements.txt
	docker build -t catalyst-base:latest . -f ./docker/Dockerfile

docker-dev: requirements.txt requirements-dev.txt
	docker build -t catalyst-dev:latest . -f ./docker/Dockerfile-dev

install-from-source:
	pip uninstall catalyst -y && pip install -e ./

clean:
	rm -rf build/
	docker rmi -f catalyst-base:latest
	docker rmi -f catalyst-dev:latest
