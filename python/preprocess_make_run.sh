#!/bin/bash

cd ode_systems 
python katz96.py $1
mv preprocess_ode.cu ../../cuda/ode_system.cu
cd ../../cuda
make clean
make
cd ../python
python python_harness.py --Ntile=$1 --fname="ntile${1}.hdf5"
