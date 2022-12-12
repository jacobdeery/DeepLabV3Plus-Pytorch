#!/usr/bin/bash

python main.py --data_root=/datasets/cityscapes \
               --dataset=cityscapes \
               --ckpt=checkpoints/baseline.pth \
               --model=deeplabv3plus_resnet101 \
               --test_only
               # --save_val_results
