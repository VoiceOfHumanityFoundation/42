#!/bin/bash
conda env create --file=environment.yml
conda activate aoi
pip3 install -r requirements.txt
