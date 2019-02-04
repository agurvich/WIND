#!/bin/bash

cd ../cuda
make clean
make
nm -D lib/wind.so > "../extra/export.txt"
cd ../python
python python_harness.py

