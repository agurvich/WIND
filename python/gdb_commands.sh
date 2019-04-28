#!/bin/bash

LINE=131
LINE2=139
cuda-gdb python -ex 'set breakpoint pending on' -ex 'dir ../cuda/BDF2' -ex "b BDF2_harness.cu:${LINE}" -ex "b BDF2_harness.cu:${LINE2}" -ex "b GDBbreakpoint" -ex 'run python_harness.py' 

### print elements of an array
#p ((@global float *)d_current_state_flat)[0]@7

#cuda-gdb python -q << EOF
#dir ../cuda/BDF2 y
#b BDF2_harness.cu:266
#run python_harness.py
#EOF

#4       breakpoint     keep y   0x00002aaac26934d3 in BDF2SolveSystem(float, float, float, float**, float*, float*, float**, float**, float*, float*, float*, float**, float*, float*, int, int) at BDF2/BDF2_harness.cu:151
        #breakpoint already hit 1 time
        #p ((@global float *)d_current_state_flat)[0]@7
        #p "after the 2/3 SIE step"
        #p ((@global float *)d_Jacobianss_flat)[0]@5
        #p ((@global float *)d_Jacobianss_flat)[5]@5
        #p ((@global float *)d_Jacobianss_flat)[10]@5
        #p ((@global float *)d_Jacobianss_flat)[15]@5
        #p ((@global float *)d_Jacobianss_flat)[20]@5
