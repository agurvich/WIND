# Add source files here
EXECUTABLE	:= invert_test.so

# Cuda source files (compiled with cudacc)
CUFILES		:= py_invert_test.cu #hello.cu kernel.cu

CFLAGS = --compiler-options '-fPIC' -shared

INCLUDE = #-I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include

LIB = -lcublas -lcusolver

## TODO how to link these to executable? 
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= matrixmul_gold.cpp

#CUDEPS		:= ODE_kernel.cu;


make: $(CUFILES) $(CUDEPS); nvcc $(CFLAGS) -arch=sm_35 $(INCLUDE) $(LIB) -o $(EXECUTABLE) $(CUFILES) $(CUDEPS)

clean: 
	rm invert_test.so *.o
