#!/bin/bash

original="/storage/users/masefi/deepFakeDetection/PCL-I2G/dataset_c40/PatchForensics/original"

python ./src/main.py \
  --real_root "$original" \
  --fake_root "/storage/users/masefi/deepFakeDetection/PCL-I2G/dataset_c40/PatchForensics/FS" \
  --exp-name train-fs-alpha-33-quality-c40-batch-16-epoch-50

python ./src/main.py \
  --real_root "$original" \
  --fake_root "/storage/users/masefi/deepFakeDetection/PCL-I2G/dataset_c40/PatchForensics/NT" \
  --exp-name train-nt-alpha-33-quality-c40-batch-16-epoch-50
