.PHONY: check-docs docker docker-fp16 docker-dev docker-dev-fp16 install-from-source clean

check-docs:
	bash ./bin/workflows/check_docs.sh

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
