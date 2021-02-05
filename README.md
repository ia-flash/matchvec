# MatchVec [![Software License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fia-flash%2Fmatchvec%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/ia-flash/matchvec/goto?ref=master)

Retourne le marque/modèle d'un véhicule à partir d'un cliché, parmi 100 classes fréquentes dans le parc automobile francais.

Obtains the make and model of a vehicle image from the most frequent cars in France.

## Installation

Build docker images

```
make build
```

Launch dockers

```
make up
```

Launch celery worker

```
make celery
```

# [Documentation](https://ia-flash.github.io/matchvec/)

Documentation is automatically generated using [Github actions](https://github.com/ia-flash/matchvec/actions) and deployed using [Github pages](https://github.com/ia-flash/matchvec/deployments).

# Utilisation

See [iaflash.fr/testapi/matchvec](https://iaflash.fr/testapi/matchvec)

# Test

`make test`

# Sam support


# License

Source code has been published using [Apache 2.0 license](LICENSE).

© 2019 Agence Nationale de Traitement Automatisé des Infractions (ANTAI), Victor Journé, Cristian Brokate
