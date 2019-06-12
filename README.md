# matchvec

Retourne le marque/modèle d'un véhicule à partir d'un cliché, parmi 100 classes fréquentes dans le parc automobile francais.

## Installation

Replace environment file for docker image

```
mv docker/env.list.sample docker/env.list
```

## Contents

```
├── docker                                 <- Docker configuration files
│   ├── conf.list
│   ├── conf.list.sample
│   ├── cpu
│   ├── env.list
│   ├── env.list.sample
│   └── gpu
├── docker-compose-gpu.yml
├── docker-compose.yml
├── docker-restart.yml
├── docs                                   <- Sphinx documentation folder
│   ├── build
│   ├── make.bat
│   ├── Makefile
│   └── source
├── Makefile                               <- Orchestring commands
├── matchvec                               <- Python application folder
│   ├── app.py
│   ├── classification.py
│   ├── __init__.py
│   ├── process.py
│   ├── retina_detection.py
│   ├── ssd_detection.py
│   ├── utils.py
│   └── yolo_detection.py
├── model                                  <- Folder for models
│   ├── resnet18-100
│   ├── ssd_mobilenet_v2_coco_2018_03_29
│   └── yolo
├── README.md                              <- Top-level README for developers using this project
└── tests                                  <- Unit test scripts
    ├── clio-peugeot.jpg
    └── test_process.py
```
