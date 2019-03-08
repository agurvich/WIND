#!/bin/bash
export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
/lib64:$LD_LIBRARY_PATH

## flags I've tried
# --events all 

NAME=SIE

cd /u/sciteam/gurvich/wind/python
rm nvvp_profiles/${NAME}*

## startup nvidia-smi in the background
device_memory_profile "nvvp_profiles/${NAME}_memory.csv" & 
nvprof -o nvvp_profiles/${NAME}.%p.out --print-gpu-trace --profile-all-processes & python python_harness.py
## kill background nvidia-smi process
killall nvidia-smi
