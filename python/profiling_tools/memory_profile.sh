#!/bin/bash
export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
/lib64:$LD_LIBRARY_PATH

NAME=${1}_${2}
DATADIR=../data/${NAME}

## NOTE this assumes SIE will always be run before RK2 is
memory_out_file="../data/${DATADIR}/${NAME}_SIE_memory.csv"
if [ -f ${memory_out_file} ]
then
    memory_out_file="../data/${DATADIR}/${NAME}_RK2_memory.csv"
fi

device_memory_profile ${memory_out_file} & 
python python_harness.py --system_name=${1} --Ntile=${2} "${@:3}"
## kill background nvidia-smi process
killall nvidia-smi
