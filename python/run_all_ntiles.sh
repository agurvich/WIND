#!/bin/bash

SYSTEM_NAME=Katz96
Ntiles=(30) #(1 5 10 15 20 25 30 40 45 50 100 200 500)
Nsystem_tiles=(50) # 5 10 20 50 100 200 500 1000) # (1) #
Nstepss=(1 5 10 15 20 25 30 40 45 50 100 200 500)

## have to recompile in fixed step mode
n_integration_steps=1

absolutes=('5e-3' '1e-3' '5e-4' '1e-4')
relatives=('5e-4' '1e-4' '5e-3' '1e-3') # 

for ABSOLUTE in "${absolutes[@]}"
do
    for RELATIVE in "${relatives[@]}"
    do
        ## overwrite and make them match, seems more stable that way
        ##RELATIVE=${ABSOLUTE}

        ## make the final directory for all the files in this grid
        maindata=../data/data_${ABSOLUTE:0:2}${ABSOLUTE:3}_${RELATIVE:0:2}${RELATIVE:3}
        if [ ! -d ${maindata} ]
            then
            mkdir ${maindata}
        fi

        ## compile the support files as well as the common files-- second true makes clean
        bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntiles[0]} ${Nsystem_tiles[0]} ${n_integration_steps} true  true ${ABSOLUTE} ${RELATIVE}
        for Ntile in "${Ntiles[@]}"
        do
            ## compile the new system for this Neqn_tile
            bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tiles[0]} ${n_integration_steps} true  false ${ABSOLUTE} ${RELATIVE}
            for Nsystem_tile in "${Nsystem_tiles[@]}"
            do
                NAME=${SYSTEM_NAME}_neqntile_${Ntile}_nsystemtile_${Nsystem_tile}_fixed_${n_integration_steps} 
                DATADIR=${maindata}/${NAME}
                if [ ! -d ${DATADIR} ]
                    then
                    bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} false false ${ABSOLUTE} ${RELATIVE}
                    echo  ${NAME} not found in ${DATADIR}
                    ## run SIE on the gpu with memory profiling
                    bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --SIE=True "${@:2}"
                    ## run RK2 on the gpu with memory profiling
                    bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --RK2=True "${@:2}"
                    ## run SIE on the cpu with memory profiling
                    bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --SIE=True --gold=True "${@:2}"
                    ## run RK2 on the cpu with memory profiling
                    bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} --RK2=True --gold=True "${@:2}"
                    ## move this file to where it belongs
                    mv ../data/${NAME} ${maindata}
                fi
            done ## for Nsystem_tile
        done ## for Ntile
    done ## for RELATIVE
done ## for ABSOLUTE
