# The Makefile defines all builds/tests steps

# include .env file
include docker/env.list

# compose command to merge production file and and dev/tools overrides
COMPOSE?=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml 

export COMPOSE
export APP_PORT

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

build:
	$(COMPOSE) build

dev:
	$(COMPOSE) up

up:
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
