#!/bin/bash
export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
/lib64:$LD_LIBRARY_PATH

NAME=${1}_${2}
DATADIR=../data/${NAME}

device_memory_profile "../data/${DATADIR}/${NAME}_memory.csv" & 
python python_harness.py --system_name=${1} --Ntile=${2} "${@:3}"
## kill background nvidia-smi process
killall nvidia-smi
