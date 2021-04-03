#!/bin/bash
# preprocess files and output images in two resolutions
python src/preprocess.py 128 $1
python src/preprocess.py 224 $1

# predict
python predict.py $2
