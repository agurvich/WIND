#!/bin/bash

LINE=223
cuda-gdb python -ex 'set breakpoint pending on' -ex 'dir ../cuda/BDF2' -ex "b BDF2_harness.cu:${LINE}" -ex "b GDBbreakpoint" -ex 'run python_harness.py' 

### print elements of an array
#p ((@global float *)d_current_state_flat)[0]@7

#cuda-gdb python -q << EOF
#dir ../cuda/BDF2 y
#b BDF2_harness.cu:266
#run python_harness.py
#EOF
