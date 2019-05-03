#!/bin/bash

LINE=419
LINE2=419
cuda-gdb debug -ex 'set breakpoint pending on' -ex 'dir ../SIE' -ex "b SIE_harness.cu:${LINE}" -ex "b SIE_harness.cu:${LINE2}" -ex "b GDBbreakpoint" -ex 'run python_harness.py' 

### print elements of an array
#p ((@global float *)d_current_state_flat)[0]@7

#cuda-gdb python -q << EOF
#dir ../cuda/SIE y
#b SIE_harness.cu:266
#run python_harness.py
#EOF

#4       breakpoint     keep y   0x00002aaac26934d3 in SIESolveSystem(float, float, float, float**, float*, float*, float**, float**, float*, float*, float*, float**, float*, float*, int, int) at SIE/SIE_harness.cu:151
        #breakpoint already hit 1 time
        #p ((@global float *)d_current_state_flat)[0]@7
        #p "after the 2/3 SIE step"
        #p ((@global float *)d_Jacobianss_flat)[0]@5
        #p ((@global float *)d_Jacobianss_flat)[5]@5
        #p ((@global float *)d_Jacobianss_flat)[10]@5
        #p ((@global float *)d_Jacobianss_flat)[15]@5
        #p ((@global float *)d_Jacobianss_flat)[20]@5