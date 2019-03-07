#!/bin/bash
export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
/lib64:$LD_LIBRARY_PATH

## flags I've tried
#--events all 

NAME=SIE

cd /u/sciteam/gurvich/wind/python
rm nvvp_profiles/${NAME}*
nvprof -o nvvp_profiles/${NAME}.%p.out --print-gpu-trace --profile-all-processes & python python_harness.py
