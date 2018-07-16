#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES
cputool --cpu-limit 30 --  python myVGGnet_test.py
