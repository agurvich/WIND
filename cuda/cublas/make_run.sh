#!/bin/bash

make clean
make
nm -D invert_test.so > "export.txt"
python python_harness.py

