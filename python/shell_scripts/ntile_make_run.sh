#!/bin/bash

    
## $1 (Katz96 or NR_test) $2 integer number of tilings
##  any other arguments are passed to wind_harness.py

## make necessary precompile files and build debug directories
bash shell_scripts/precompile_system.sh "${@:1}"

## actually run the thing
#python wind_harness.py  --system_name=${1} --Ntile=${2} --Nsystem_tile=${3} --n_integration_steps=${4} --absolute=${5} --relative=${6} "${@:7}"
