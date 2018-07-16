#! /bin/bash
export DISPLAY=:1
export XAUTHORITY=~/.Xauthority
#nvidia-settings -a [gpu:2]/GPUfanControlState=1
#nvidia-settings -a [fan:1]/GPUTargetFanSpeed=50

nvidia-settings -a [gpu:2]/GPUFanControlState=1 -a [fan:2]/GPUTargetFanSpeed=50
