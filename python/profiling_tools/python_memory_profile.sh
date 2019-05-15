#!/bin/bash

NAME=${1}_neqntile_${2}_nsystemtile_${3}_fixed_${4}
DATADIR=../data/${NAME}

## NOTE this assumes SIE will always be run before RK2 is
memory_out_file="../data/${DATADIR}/${NAME}_SIEgold_memory.dat"
if [ -f ${memory_out_file} ]
then
    memory_out_file="../data/${DATADIR}/${NAME}_RK2gold_memory.dat"
fi

mprof run --interval 0.01 --output ${memory_out_file} wind_harness.py --system_name=${1} --Ntile=${2} --Nsystem_tile=${3} --n_integration_steps=${4} "${@:5}"
