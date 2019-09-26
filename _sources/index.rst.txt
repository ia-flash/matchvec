.. IAFlash documentation master file, created by
   sphinx-quickstart on Mon May 13 14:03:07 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IAFlash's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

::

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

Contents
========

Application modules

Main app
--------

.. automodule:: app
    :members:

Process function
----------------

.. automodule:: process
    :members:


Classification model
--------------------

.. automodule:: classification_onnx
    :members:


.. automodule:: classification_torch
    :members:


Detection avec SSD
------------------

.. automodule:: ssd_detection
    :members:


Detection avec Yolo
-------------------

.. automodule:: yolo_detection
    :members:


Other functions
---------------

.. automodule:: utils
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
