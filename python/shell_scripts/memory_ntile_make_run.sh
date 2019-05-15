#!/bin/bash

    
## $1 (Katz96 or NR_test) $2 integer number of tilings
##  any other arguments are passed to python_harness.py

## make necessary precompile files and build
bash shell_scripts/precompile_system.sh "${@:1}"

## actually run the thing but with the memory profiler attached
#bash profiling_tools/memory_profile.sh "${@:1}"


