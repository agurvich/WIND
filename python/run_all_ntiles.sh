#!/bin/bash

SYSTEM_NAME=Katz96
Ntiles=(1 5 10 15 20 25 30 40 45 50 100 200 500)
Nsystem_tiles=(1) #(1 5 10 15 20 25 30 40 45 50 100 200 500)
Nstepss=(1 5 10 15 20 25 30 40 45 50 100 200 500)

## have to recompile in fixed step mode
n_integration_steps=1


for Ntile in "${Ntiles[@]}"
do
    for Nsystem_tile in "${Nsystem_tiles[@]}"
    do
        NAME=${SYSTEM_NAME}_neqntile_${Ntile}_nsystemtile_${Nsystem_tile}_fixed_${n_integration_steps}
        DATADIR=../data/${NAME}
        if [ ! -d ${DATADIR} ]
            then
            ## only need to compile for different tilings that affect # of equations/system
            bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps}
            echo  ${NAME} not found!
            bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --SIE=True "${@:2}"
            bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --RK2=True "${@:2}"
            bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --SIE=True --gold=True "${@:2}"
            bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --RK2=True --gold=True "${@:2}"
        fi
    done
done
