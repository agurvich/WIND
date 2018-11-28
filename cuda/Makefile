# Add source files here
EXECUTABLE	:= arradd.so

# Cuda source files (compiled with cudacc)
CUFILES		:= hello.cu

CFLAGS = -shared

## TODO how to link these to executable? 
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= matrixmul_gold.cpp

#CUDEPS		:= ODE_kernel.cu;


hellomake: $(CUFILES) $(CUDEPS); nvcc --compiler-options '-fPIC' $(CFLAGS) -o $(EXECUTABLE) $(CUFILES) $(CUDEPS) 

clean: 
	rm arradd *.o
