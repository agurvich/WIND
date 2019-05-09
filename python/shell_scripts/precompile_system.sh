#!/bin/bash

DATADIR=../data/${1}_nsystem_1_${2}

echo "Compiling files for" $1 "with" $2 "tiling"
python make_ode_cu.py --system_name=${1} --Ntile=${2} "${@:3}"

cp `pwd`/${DATADIR}/$1_nsystem_1_$2_preprocess_ode.cu ../cuda/ode_system.cu
cp `pwd`/${DATADIR}/$1_nsystem_1_$2_preprocess_RK2_kernel.cu ../cuda/RK2/kernel.cu
cd ../cuda
pwd
## build the new ODE system and link to the solver .so 's
#make
cd ../python
