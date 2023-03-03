all:
	docker login docker.yfish.x
	docker pull docker.yfish.x/yfish-pytorch-lightweight
	docker build --tag docker.yfish.x/fergie:v3.0.0 .
	docker tag docker.yfish.x/fergie:v3.0.0 docker.yfish.x/fergie:latest
	docker push docker.yfish.x/fergie:latest
	docker push docker.yfish.x/fergie:v3.0.0
