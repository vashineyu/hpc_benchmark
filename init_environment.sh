#!/bin/sh

conda env create -q -f ./env_tf_keras.yaml
source activate tf_keras
pip install horovod
pip install imgaug
