#!/bin/bash

python make_ode_cu.py --system_name="katz" --Ntile=$1
cp precompile_cu_files/Katz96_preprocess_ode.cu ../cuda/ode_system.cu
cd ../cuda
pwd
make clean
make
cd ../python
python python_harness.py --Ntile=$1 --katz=True --NR=False --SIE=True --PY=False
