#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES
cputool --cpu-limit 80 --  python myVGGnet_train.py
