.PHONY: check-docs docker docker-fp16 docker-dev docker-dev-fp16 install-from-source clean

check-docs:
	bash ./bin/codestyle/_check_docs.sh

docker: ./requirements/
	echo building $${REPO_NAME:-catalyst-base}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst-base}:$${TAG:-latest} . \
		-f ./docker/Dockerfile --no-cache

docker-fp16: ./requirements/
	echo building $${REPO_NAME:-catalyst-base-fp16}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst-base-fp16}:$${TAG:-latest} . \
		-f ./docker/Dockerfile-fp16 --no-cache

docker-dev: ./requirements/
	echo building $${REPO_NAME:-catalyst-dev}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst-dev}:$${TAG:-latest} . \
		-f ./docker/Dockerfile-dev --no-cache

docker-dev-fp16: ./requirements/
	echo building $${REPO_NAME:-catalyst-dev-fp16}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst-dev-fp16}:$${TAG:-latest} . \
		-f ./docker/Dockerfile-dev-fp16 --no-cache

install-from-source:
	pip uninstall catalyst -y && pip install -e ./

clean:
	rm -rf build/
	docker rmi -f catalyst-base:latest
	docker rmi -f catalyst-base-fp16:latest
	docker rmi -f catalyst-dev:latest
	docker rmi -f catalyst-dev-fp16:latest


run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=catalystdev \
		-v $(shell pwd):/workspace/ catalystteam/catalyst:20.03-dev-fp16 bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it catalystdev bash

stop: ## Stop and remove a running container
	docker stop catalystdev; docker rm catalystdev