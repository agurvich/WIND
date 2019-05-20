#!/bin/bash
#export LD_LIBRARY_PATH=$CRAY_CUDATOOLKIT_DIR
#/lib64:$LD_LIBRARY_PATH

## NOTE this assumes SIE will always be run before RK2 is
memory_out_file="../data/${DATADIR}/SIE_memory.csv"
if [ -f ${memory_out_file} ]
then
    memory_out_file="../data/${DATADIR}/RK2_memory.csv"
fi

device_memory_profile ${memory_out_file} & 
python wind_harness.py  --system_name=${1} --Ntile=${2} --Nsystem_tile=${3} --n_integration_steps=${4} --absolute=${5} --relative=${6} "${@:7}"
## kill background nvidia-smi process
killall nvidia-smi
