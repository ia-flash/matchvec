# The Makefile defines all builds/tests steps

# include .env file
include docker/conf.list

# compose command for dev env
COMPOSE?=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml

# compose command for prod env
#COMPOSE?=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml -f docker-restart.yml

export COMPOSE
export APP_PORT
export MODEL_PATH


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

build:
	$(COMPOSE) build

dev: docker/env.list docker/conf.list
	$(COMPOSE) up

up: docker/env.list docker/conf.list
	$(COMPOSE) up -d

stop:
	$(COMPOSE) stop

exec:
	$(COMPOSE) exec matchvec bash

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f --tail 50

docs/html:
	$(COMPOSE) exec matchvec make -C /app/docs html

docs: docs/html
	echo "Post"

test:
	$(COMPOSE) exec matchvec python tests/test_process.py
