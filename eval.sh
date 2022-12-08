#!/usr/bin/bash

python main.py --data_root=/datasets/wd_public_02 \
               --dataset=wilddash \
               --ckpt=checkpoints/baseline.pth \
               --model=deeplabv3plus_resnet101 \
               --test_only \
               --save_val_results
