#!/bin/bash

SYSTEM_NAME=$1
Ntiles=(1 5 10 15 20 25 30 40 45 50 100 200 500)

for Ntile in "${Ntiles[@]}"
do
    NAME=${SYSTEM_NAME}_${Ntile}
    DATADIR=../data/${NAME}
    if [ ! -d ${DATADIR} ]
        then
        echo  ${NAME} not found!
        bash shell_scripts/memory_ntile_make_run.sh ${SYSTEM_NAME} ${Ntile} --SIE=True "${@:2}"
        bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} --RK2=True "${@:2}"
    fi
done
