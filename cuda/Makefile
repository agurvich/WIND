# Add source files here
EXECUTABLE	:= arradd.so

# Cuda source files (compiled with cudacc)
CUFILES		:= hello.cu kernel.cu

CFLAGS = -shared

## TODO how to link these to executable? 
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= matrixmul_gold.cpp

#CUDEPS		:= ODE_kernel.cu;


make: $(CUFILES) $(CUDEPS); nvcc --compiler-options '-fPIC' $(CFLAGS) -arch=sm_35 -o $(EXECUTABLE) $(CUFILES) $(CUDEPS)

clean: 
	rm arradd.so *.o
