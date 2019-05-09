#!/bin/bash

    
## $1 (Katz96 or NR_test) $2 integer number of tilings
##  any other arguments are passed to python_harness.py

## make necessary precompile files and build
bash shell_scripts/precompile_system.sh $1 $2 $4

python python_harness.py  --system_name=$1 --Ntile=$2 "${@:3}"

#bash shell_scripts/move_debug_file.sh $1 $2 

