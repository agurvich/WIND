#!/bin/bash

python make_ode_cu.py --system_name="NR" --Ntile=$1
cp precompile_cu_files/NR_test_preprocess_ode.cu ../cuda/ode_system.cu
cd ../cuda
pwd
make clean
make
cd ../python
python python_harness.py --Ntile=$1 --NR=True --katz=False --SIE=True --PY=False
