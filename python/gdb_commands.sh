#!/bin/bash

LINE=266
cuda-gdb python -ex 'set breakpoint pending on' -ex 'dir ../cuda/BDF2' -ex "b BDF2_harness.cu:${LINE}" -ex 'run python_harness.py'

#cuda-gdb python -q << EOF
#dir ../cuda/BDF2 y
#b BDF2_harness.cu:266
#run python_harness.py
#EOF
