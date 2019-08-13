#!/bin/bash

SYSTEM_NAME=StiffTrig #Katz96
Ntiles=(75) #1 5 10 15 20 25 30 40 45 50 100 200 500)
Nsystem_tiles=(20) # 5 10 20 50 100 200 500 1000) # (1) #

## have to recompile in fixed step mode
n_integration_steps=1

absolutes=('5e-3' ) #'1e-3' '5e-4' '1e-4'
relatives=('5e-3' ) # '1e-3' '5e-4' '1e-4' 

## TODO this must be ../data, should fix that...
maindata=../data
## make the final directory for all the files in this grid
if [ ! -d ${maindata} ]
    then
    mkdir ${maindata}
fi

function changenamedatadir(){
    ind=`echo ${5} | sed -n 's/[-].*//p' | wc -c`
    abs_string=${5:0:ind-1}${5:ind}
    ind=`echo ${6} | sed -n 's/[-].*//p' | wc -c`
    rel_string=${6:0:ind-1}${6:ind}
    export NAME=${1}_neqntile.${2}_nsystemtile.${3}_fixed.${4}_abs.${abs_string}_rel.${rel_string}
    export DATADIR=${maindata}/${NAME}
}

for Ntile in "${Ntiles[@]}"
do
    echo "Making a tempory directory to produce precompile files..."
    ## 2 -> dummy value for ${n_integration_steps} that should never overlap
    changenamedatadir ${SYSTEM_NAME} ${Ntile} ${Nsystem_tiles[0]} 2 ${absolutes[0]} ${relatives[0]}
    ## compile the new system for this Neqn_tile
    bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tiles[0]} 2 ${absolutes[0]} ${relatives[0]} true true
    ## i feel a little better by moving it before automated-ly rm'ing it
    mv ${DATADIR} ${maindata}/trash
    rm -r ${maindata}/trash
    echo "...done with temporary directory"

    for ABSOLUTE in "${absolutes[@]}"
    do
        for RELATIVE in "${relatives[@]}"
        do
            for Nsystem_tile in "${Nsystem_tiles[@]}"
            do
                changenamedatadir ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE}
                if [ ! -d ${DATADIR} ]
                    then
                    echo  ${NAME} not found in ${maindata}
                    ## create the datadir
                    #bash shell_scripts/precompile_system.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE} false false 
                    ## run SIE on the gpu with memory profiling
                    #bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE} --SIE=True "${@:1}"
                    ## run RK2 on the gpu with memory profiling
                    #bash profiling_tools/memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE} --RK2=True "${@:1}"
                    ## run SIE on the cpu with memory profiling
                    #bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE} --SIE=True --gold=True "${@:1}"
                    ## run RK2 on the cpu with memory profiling
                    #bash profiling_tools/python_memory_profile.sh ${SYSTEM_NAME} ${Ntile} ${Nsystem_tile} ${n_integration_steps} ${ABSOLUTE} ${RELATIVE} --RK2=True --gold=True "${@:1}"
                else
                    echo "Nothing to do for" ${NAME}
                fi
            done ## for Nsystem_tile
        done ## for RELATIVE
    done ## for ABSOLUTE
done ## for Ntile
