#!/bin/bash

python bo.py \
  --data-name eq_structures \
  --save-appendix DVAE_EQ \
  --checkpoint 300 \
  --res-dir="EQ_results/" \
  --BO-rounds 10 \
  --BO-batch-size 50 \
  --random-as-test \
  --random-as-train \
  --random-baseline