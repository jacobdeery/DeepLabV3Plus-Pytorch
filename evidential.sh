#!/usr/bin/bash

python main.py --data_root=/datasets/cityscapes \
               --dataset=cityscapes \
               --loss_type=evidential \
               --model=evidentialdeeplab_resnet101 \
               --batch_size=8 \
               --val_interval=500 \
               --ckpt=checkpoints/latest_evidentialdeeplab_resnet101_cityscapes_os16.pth --continue_training \
               # --test_only \
               # --save_val_results \
