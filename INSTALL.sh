#!/usr/bin/bash
conda env create --file=environment.yml
conda activate aoi
pip install -r requirements.txt
