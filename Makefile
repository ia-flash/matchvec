# The Makefile defines all builds/tests steps

# include .env file
include docker/conf.list

export GPU

ifeq ($(GPU),1)
		export COMPOSE=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml -f docker-compose-gpu.yml
endif

# compose command for dev env
COMPOSE?=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml
# compose command for prod env
ifeq ($(RESTART),1)
	COMPOSE += -f docker-restart.yml
endif

export

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

docker/env.list:
	# Copy default config
	cp docker/env.list.sample docker/env.list

docker/conf.list:
	# Copy default config
	cp docker/conf.list.sample docker/conf.list

network:
	@docker network create isolated_nw 2> /dev/null; true

build:
	$(COMPOSE) build

dev: docker/env.list docker/conf.list network
	$(COMPOSE) up

up: docker/env.list docker/conf.list network
	$(COMPOSE) up -d

celery:
	if [ "${GPU}" = 1 ]; then \
	$(COMPOSE) exec matchvec celery worker -A app.celery --loglevel=debug --workdir /app/matchvec --pool solo; \
	else \
	$(COMPOSE) exec matchvec celery worker -A app.celery --loglevel=debug --workdir /app/matchvec; \
	fi

celery_prod:
	if [ "${GPU}" = 1 ]; then \
	$(COMPOSE) exec matchvec celery worker -A app.celery --loglevel=error --workdir /app/matchvec --pool solo --detach; \
	else \
	$(COMPOSE) exec matchvec celery worker -A app.celery --loglevel=error --workdir /app/matchvec --detach; \
	fi

stop:
	$(COMPOSE) stop

exec:
	$(COMPOSE) exec matchvec bash

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f --tail 50

docs/html:
	$(COMPOSE) exec matchvec python matchvec/export_swagger.py
	$(COMPOSE) exec matchvec make -C /app/docs html

docs: docs/html
	echo "Post"

test:
	#You can test a specific function with for instance:
	# make test test=test_process.py::test_class_prio
	$(COMPOSE) exec matchvec pytest tests/$(test) -s

layers:
	mkdir -p layers/opencv/python
	mkdir -p layers/onnxruntime/python
	mkdir -p layers/pillow/python

layer: layers
	$(COMPOSE) exec matchvec pip3 install opencv-python-headless==4.0.0.21 -t layers/opencv/python
	$(COMPOSE) exec matchvec pip3 install onnxruntime==1.5.1 -t layers/onnxruntime/python
	$(COMPOSE) exec matchvec pip3 install Pillow==6.1.0 requests-toolbelt==0.9.1 -t layers/pillow/python
	cd layers/opencv; zip -r opencv.zip python; cd ../..;
	cd layers/onnxruntime; zip -r onnxruntime.zip python; cd ../..;
	cd layers/pillow; zip -r pillow.zip python; cd ../..;

layer_matchvec:
	mkdir -p layers/matchvec/python
	$(COMPOSE) exec matchvec python setup.py build -b layers/matchvec
	cd layers/matchvec;cp lib/matchvec python -r;zip -r matchvec.zip python; cd ../..;
	aws lambda publish-layer-version --layer-name matchvec --zip-file fileb://layers/matchvec/matchvec.zip --compatible-runtimes python3.6



layer_publish:
		aws lambda publish-layer-version --layer-name onnxruntime --zip-file fileb://layers/onnxruntime/onnxruntime.zip --compatible-runtimes python3.6
		aws lambda publish-layer-version --layer-name opencv --zip-file fileb://layers/opencv/opencv.zip --compatible-runtimes python3.6
		aws lambda publish-layer-version --layer-name pillow --zip-file fileb://layers/pillow/pillow.zip --compatible-runtimes python3.6

sam_build:
	rm -rf aws_lambda/matchvec
	cp matchvec aws_lambda -r;cd aws_lambda;rm -rf __pycache__;sam build

sam_local:
	sam local start-api

##########
#  Test  #
##########

# Curl test
# curl -X POST -d body@tests/clio4.jpg  http://localhost:3000/post

# Generate event
# sam local generate-event apigateway aws-proxy --body "" --path "post" --method GET > api-event.json
# Use event to invoke api
# sam local invoke  -e api-event.json

sam_package:
	sam package --template-file aws_lambda/template.yaml --s3-bucket iaflash --output-template-file aws_lambda/packaged-anonym.yaml

sam_deploy:
	aws cloudformation delete-stack --stack-name matchvec-anonym;sleep 15;\
	aws cloudformation deploy --template-file aws_lambda/packaged-anonym.yaml --stack-name matchvec-anonym
	aws apigateway get-rest-apis

# test aws lambda
# curl -F image=@clio4.jpg -F "url=https://upload.wikimedia.org/wikipedia/commons/3/31/Renault_Clio_front_20080116.jpg" https://p4veiq2ftb.execute-api.eu-west-1.amazonaws.com/Prod/predict
