.PHONY: check-style check-docs docker docker-dev clean

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	pre-commit run

check-docs:
	bash ./bin/_check_docs.sh

docker: ./requirements/
	docker build -t catalyst-base:latest . -f ./docker/Dockerfile

docker-fp16: ./requirements/
	docker build -t catalyst-base-fp16:latest . -f ./docker/Dockerfile-fp16

docker-dev: ./requirements/
	docker build -t catalyst-dev:latest . -f ./docker/Dockerfile-dev

docker-dev-fp16: ./requirements/
	docker build -t catalyst-dev-fp16:latest . -f ./docker/Dockerfile-dev-fp16

install-from-source:
	pip uninstall catalyst -y && pip install -e ./

clean:
	rm -rf build/
	docker rmi -f catalyst-base:latest
	docker rmi -f catalyst-base-fp16:latest
	docker rmi -f catalyst-dev:latest
	docker rmi -f catalyst-dev-fp16:latest
