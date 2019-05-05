#!/bin/bash

python make_ode_cu.py --system_name="NR" --Ntile=$1
cp precompile_cu_files/NR_test_${1}_preprocess_ode.cu ../cuda/ode_system.cu
cd ../cuda
pwd
## build the new ODE system and link to the solver .so 's
make
cd ../python
#python python_harness.py --Ntile=$1 --katz=False --NR=True --SIE=True --PY=False
cp ../data/NR_test_${1}_debug.txt ../cuda/debug/input${1}.h
echo "#"include '"'input${1}.h'"' > ../cuda/debug/debug.c
less ../cuda/debug/debug_suffix.c >> ../cuda/debug/debug.c
cd ../cuda/debug
make
