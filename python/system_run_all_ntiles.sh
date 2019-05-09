#!/bin/bash

SYSTEM_NAME=$1
Ntiles=(5 10 15 20 25 30 40 45 50 100 200 500)

for Ntile in "${Ntiles[@]}"
do
    NAME=${SYSTEM_NAME}_fixed_${Ntile}_1
    DATADIR=../data/${NAME}
    if [ ! -d ${DATADIR} ]
        then
        echo  ${NAME} not found!
        bash shell_scripts/memory_ntile_make_run.sh ${SYSTEM_NAME} 1 "_fixed_${Ntile}" --nsteps=${Ntile} "${@:2}" --SIE=True  --makeplots=False  
        bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} 1 "_fixed_${Ntile}" --nsteps=${Ntile} "${@:2}" --RK2=True --makeplots=False  
    fi
done
