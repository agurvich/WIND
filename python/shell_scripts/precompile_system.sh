#!/bin/bash

echo "Compiling files for" $1 "with" $2 "eqn tiling" $3 "system tiling" $4 "integration steps" $5 "absolute" $6 "relative"
python wind_harness.py --system_name=${1} --Ntile=${2} --Nsystem_tile=${3} --n_integration_steps=${4} --absolute=${5} --relative=${6} --dumpDebug=True # -dumpDebug=True "${@:5}"

## passed flag whether we should rebuild or not
if [ ${7} == true ]
    then
    cp `pwd`/${DATADIR}/precompile_device_dydt.cu ../cuda/ode_system
    cp `pwd`/${DATADIR}/precompile_ode_system.cu ../cuda/ode_system
    cp `pwd`/${DATADIR}/precompile_ode_gold.c ../cuda/ode_system
    cd ../cuda
    pwd
    if [ ${8} == true ]
        then
        echo "Making clean..."
        make clean > /dev/null 2>&1
        make preclean > /dev/null 2>&1
        echo "...done"
    fi 
    ## build the new ODE system and link to the solver .so 's
    echo "Making..."
    make ODEOBJS="ode_system/precompile_ode_system.o ode_system/precompile_device_dydt.o" GOLDODEOBJS="../ode_system/precompile_ode_gold.o" > /dev/null #2>&1
    echo "...done"
    cd ../python
fi

## move the debug file that's generated to the corresponding debug directories
##  and make those
bash shell_scripts/move_debug_file.sh "${@:1}"
