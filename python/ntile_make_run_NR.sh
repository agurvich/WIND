#!/bin/bash

python make_ode_cu.py --system_name="NR" --Ntile=$1
cp precompile_cu_files/NR_test_${1}_preprocess_ode.cu ../cuda/ode_system.cu
cd ../cuda
pwd
## build the new ODE system and link to the solver .so 's
make
cd ../python
python python_harness.py --Ntile=$1 --NR=True --katz=False --SIE=True --PY=False
