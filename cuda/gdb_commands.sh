#!/bin/bash

LINE=36
LINE2=240
FILE1=main.cu
FILE2=device_dydt.cu

cuda-gdb sie_main -ex 'set breakpoint pending on' -ex 'dir RK2' -ex 'dir common' -ex 'dir CHIMES_TEMP' -ex 'dir SIE' -ex "b ${FILE1}:${LINE}" -ex "b ${FILE2}:${LINE2}"

#-ex 'dir ../c_baseline/rk2'

#-ex "command 1" -ex "p ((@global float *) d_inversess_flat)[0]@5" -ex "p ((@global float *) d_inversess_flat)[5]@5" -ex "p ((@global float *) d_inversess_flat)[10]@5" -ex "p ((@global float *) d_inversess_flat)[15]@5" -ex "p ((@global float *) d_inversess_flat)[20]@5" -ex "end"

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
