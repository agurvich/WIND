# Add source files here
EXECUTABLE	:= arradd.so

# Cuda source files (compiled with cudacc)
CUFILES		:= hello.cu kernel.cu

CFLAGS = -shared

INCLUDE = -I/sw/xe/magma/1.5.0/cnl5.2_gnu4.8.2/include

## TODO how to link these to executable? 
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= matrixmul_gold.cpp

#CUDEPS		:= ODE_kernel.cu;


make: $(CUFILES) $(CUDEPS); nvcc --compiler-options '-fPIC' $(CFLAGS) -arch=sm_35 $(INCLUDE) -o $(EXECUTABLE) $(CUFILES) $(CUDEPS)

clean: 
	rm arradd.so *.o
