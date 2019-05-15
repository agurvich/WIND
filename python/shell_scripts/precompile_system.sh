#!/bin/bash

NAME=${1}_neqntile_${2}_nsystemtile_${3}_fixed_${4}
DATADIR=../data/${NAME}

echo "Compiling files for" $1 "with" $2 "eqn tiling" $3 "system tiling" $4 "integration steps"
python python_harness.py --system_name=${1} --Ntile=${2} --Nsystem_tile=${3} --n_integration_steps=${4} #"${@:5}"

cp `pwd`/${DATADIR}/precompile_device_dydt.cu ../cuda/ode_system
cp `pwd`/${DATADIR}/precompile_ode_system.cu ../cuda/ode_system
cp `pwd`/${DATADIR}/precompile_ode_gold.c ../cuda/ode_system
cd ../cuda
pwd
## build the new ODE system and link to the solver .so 's
make ODEOBJS="ode_system/precompile_ode_system.o ode_system/precompile_device_dydt.o" GOLDODEOBJS="../ode_system/precompile_ode_gold.o"
cd ../python

## move the debug file that's generated to the corresponding debug directories
##  and make those
bash shell_scripts/move_debug_file.sh "${@:1}"
