#!/bin/bash

SYSTEM_NAME=$1
Ntiles=(1 5 10 25 40 75 125 300 500 800 1000 3000 10000)

for Ntile in "${Ntiles[@]}"
do
    NAME=${SYSTEM_NAME}_nsystem_${Ntile}_40
    DATADIR=../data/${NAME}
    if [ ! -d ${DATADIR} ]
        then
        echo  ${NAME} not found!
        bash shell_scripts/memory_ntile_make_run.sh ${SYSTEM_NAME} 40 "_nsystem_${Ntile}" --Nsystem_tile=${Ntile} "${@:2}" --SIE=True  --makeplots=False  
        bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} 40 "_nsystem_${Ntile}" --Nsystem_tile=${Ntile} "${@:2}" --RK2=True --makeplots=False  
        #python python_harness.py --system_name=${SYSTEM_NAME} --Ntile=1 --nsteps=${Ntile} "${@:2}" --RK2=True --makeplots=False  
    fi
done
