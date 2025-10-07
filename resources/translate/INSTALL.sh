#!/usr/bin/bash
conda env create --file=environment.yml
conda activate trans
pip3 install -r requirements.txt
bash run.sh
