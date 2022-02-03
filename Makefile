.PHONY: check-docs docker docker-dev install-from-source clean

PYTHON ?= python
TAG=$(shell ${PYTHON} docker/collect_dependencies_hash.py)

check-docs:
	bash ./bin/workflows/check_docs.sh

docker:
	echo building $${REPO_NAME:-catalyst}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst}:$${TAG:-latest} . \
		-f ./docker/Dockerfile --no-cache \
		--build-arg PYTORCH_TAG=$$PYTORCH_TAG \
		--build-arg CATALYST_DEV=$$CATALYST_DEV \
		--build-arg CATALYST_CV=$$CATALYST_CV \
		--build-arg CATALYST_ML=$$CATALYST_ML \
		--build-arg CATALYST_OPTUNA=$$CATALYST_OPTUNA \
		--build-arg CATALYST_ONNX=$$CATALYST_ONNX \
		--build-arg CATALYST_ONNX_GPU=$$CATALYST_ONNX_GPU

docker-dev:
	echo building $${REPO_NAME:-catalyst-dev}:$${TAG:-latest} ...
	docker build \
		-t $${REPO_NAME:-catalyst-dev}:$${TAG:-latest} . \
		-f ./docker/Dockerfile --no-cache \
		--build-arg PYTORCH_TAG=$$PYTORCH_TAG \
		--build-arg CATALYST_DEV=$$CATALYST_DEV \
		--build-arg CATALYST_CV=$$CATALYST_CV \
		--build-arg CATALYST_ML=$$CATALYST_ML \
		--build-arg CATALYST_OPTUNA=$$CATALYST_OPTUNA \
		--build-arg CATALYST_ONNX=$$CATALYST_ONNX \
		--build-arg CATALYST_ONNX_GPU=$$CATALYST_ONNX_GPU

install-from-source:
	pip uninstall catalyst -y && pip install -e ./

clean:
	rm -rf build/
	docker rmi -f catalyst:latest
	docker rmi -f catalyst-dev:latest

check:
	catalyst-make-codestyle -l 89
	catalyst-check-codestyle -l 89

