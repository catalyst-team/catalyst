.PHONY: docker clean

docker:
	docker build -t catalyst-base:latest . -f ./docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-base:latest
