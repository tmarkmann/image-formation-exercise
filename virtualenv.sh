#!/bin/bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install numpy opencv-contrib-python
deactivate
