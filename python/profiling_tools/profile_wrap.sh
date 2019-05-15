#!/bin/bash
export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
/lib64:$LD_LIBRARY_PATH

## flags I've tried
# --events all 

NAME=SIE
Nsystems="1e8"

cd /u/sciteam/gurvich/wind/python
rm nvvp_profiles/${NAME}_${Nsystems}*out

## startup nvidia-smi in the background
device_memory_profile "nvvp_profiles/${NAME}_${Nsystems}_memory.csv" & 
nvprof -o nvvp_profiles/${NAME}_${Nsystems}.%p.out --print-gpu-trace --profile-all-processes & 
python wind_harness.py --Nsystems=${Nsystems}
## kill background nvidia-smi process
killall nvidia-smi
