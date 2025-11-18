#!/bin/bash

original="/mnt/d/Datasets/FF++/PatchForensics/original"

python ./src/main.py \
  --real_root "$original" \
  --fake_root "/mnt/d/Datasets/FF++/PatchForensics/DF" \
  --norm "imagenet"
