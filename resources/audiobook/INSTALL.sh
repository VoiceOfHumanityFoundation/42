#!/usr/bin/bash
conda env create --file=environment.yml
conda activate indexTTS
pip3 install -r requirements.txt
