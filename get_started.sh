#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

# mkdir -p $EXP_DIR
#
# # Creates the environment
# conda create -n tf python=3.6
#
# # Activates the environment
# source activate tf

# pip install into environment
pip3 install -r requirements.txt

conda install pytorch -c pytorch

# download punkt and perluniprops
python -m nltk.downloader punkt
python -m nltk.downloader perluniprops

# download and preprocess SQuAD data and save in data/
# mkdir -p "$DATA_DIR"
# rm -rf "$DATA_DIR"
# python "$CODE_DIR/preprocessing/squad_preprocess.py" --data_dir "$DATA_DIR"

# Download GloVe vectors to data/
# python "$CODE_DIR/preprocessing/download_wordvecs.py" --download_dir "$DATA_DIR"
