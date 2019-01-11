#!/bin/bash

cd ../cuda
make clean
make
nm -D arradd.so > "export.txt"
cd ../python
python python_harness.py

